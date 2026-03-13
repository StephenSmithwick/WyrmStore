use crate::config::Config;
use crate::embed::{build_embedder, cosine_similarity, EmbedderApi};
use crate::store::TopicStore;
use crate::types::{
    ChunkRecord, ChunkStatus, CorrectiveAction, DaemonRequest, DaemonResponse, InjectedChunk,
    InjectedChunkInfo, MemoryIn, MemoryOut, Message, ScoredChunk, Signal, SignalType,
};
use std::collections::{HashMap, HashSet};
use std::io;
use std::sync::OnceLock;
use std::time::{SystemTime, UNIX_EPOCH};
use tiktoken_rs::{cl100k_base, CoreBPE};
use uuid::Uuid;

#[derive(Default)]
struct HotBufferState {
    messages: Vec<Message>,
    token_count: usize,
    last_conversation_hashes: Vec<u64>,
}

pub struct Daemon {
    config: Config,
    embedder: Box<dyn EmbedderApi>,
    topics: HashMap<String, TopicStore>,
    hot_buffers: HashMap<String, HotBufferState>,
    recent_injected: HashMap<String, HashMap<Uuid, InjectedChunkInfo>>,
}

impl Daemon {
    pub fn new(config: Config) -> io::Result<Self> {
        let embedder = build_embedder(&config)?;
        Ok(Self::new_with_embedder(config, embedder))
    }

    pub fn new_with_embedder(config: Config, embedder: Box<dyn EmbedderApi>) -> Self {
        Self {
            config,
            embedder,
            topics: HashMap::new(),
            hot_buffers: HashMap::new(),
            recent_injected: HashMap::new(),
        }
    }

    pub fn handle_request(&mut self, request: DaemonRequest) -> io::Result<DaemonResponse> {
        let memory_in = request.memory_in.clone().unwrap_or_else(|| MemoryIn {
            topic_id: None,
            related_topic_ids: None,
            corrections: None,
        });
        let topic_id = memory_in
            .topic_id
            .clone()
            .unwrap_or_else(|| "default".to_string());
        let related_topic_ids = memory_in.related_topic_ids.clone().unwrap_or_default();

        self.ensure_topic(&topic_id)?;
        for related in &related_topic_ids {
            self.ensure_topic(related)?;
        }

        let mut signals: Vec<Signal> = Vec::new();
        if let Some(corrections) = memory_in.corrections.as_ref() {
            let mut correction_signals = self.apply_corrections(&topic_id, corrections)?;
            signals.append(&mut correction_signals);
        }

        self.ingest_messages(&topic_id, &request.messages)?;

        let mut results = self.retrieve(&topic_id, &related_topic_ids, &request.messages)?;

        let (selected, pressure_signal, overflow_signal) =
            apply_context_budget(&self.config, &mut results, &topic_id);
        if let Some(signal) = pressure_signal {
            signals.push(signal);
        }
        if let Some(signal) = overflow_signal {
            signals.push(signal);
        }

        let mut injected_chunks = Vec::new();
        let mut injected_map = HashMap::new();
        let mut memory_lines = Vec::new();

        for chunk in selected {
            injected_chunks.push(InjectedChunk {
                id: chunk.record.id,
                topic_id: chunk.topic_id.clone(),
                canonical_id: chunk.record.canonical_id,
            });
            injected_map.insert(
                chunk.record.id,
                InjectedChunkInfo {
                    topic_id: chunk.topic_id.clone(),
                    canonical_id: chunk.record.canonical_id,
                },
            );
            memory_lines.push(format!("[mem:{}] {}", chunk.record.id, chunk.record.content));
        }

        self.recent_injected.insert(topic_id.clone(), injected_map);

        let mut messages = request.messages.clone();
        if !memory_lines.is_empty() {
            let injection = Message {
                role: "developer".to_string(),
                content: memory_lines.join("\n"),
                tool_calls: None,
                extra: Default::default(),
            };
            insert_before_last_user(&mut messages, injection);
        }

        let memory_out = MemoryOut {
            injected_chunks,
            signals: if signals.is_empty() { None } else { Some(signals) },
        };

        Ok(DaemonResponse { messages, memory_out })
    }

    fn ensure_topic(&mut self, topic_id: &str) -> io::Result<()> {
        if !self.topics.contains_key(topic_id) {
            let store = TopicStore::load(&self.config, topic_id)?;
            self.topics.insert(topic_id.to_string(), store);
        }
        Ok(())
    }

    fn ingest_messages(&mut self, topic_id: &str, messages: &[Message]) -> io::Result<()> {
        {
            let hot = self.hot_buffers.entry(topic_id.to_string()).or_default();

            let incoming_hashes: Vec<u64> = messages.iter().map(message_digest).collect();
            let lcp = longest_common_prefix(&hot.last_conversation_hashes, &incoming_hashes);

            if lcp == 0 && !hot.last_conversation_hashes.is_empty() {
                hot.messages.clear();
                hot.token_count = 0;
            }

            hot.last_conversation_hashes = incoming_hashes;

            let new_messages = messages.iter().skip(lcp).cloned().collect::<Vec<_>>();
            for message in new_messages {
                if !is_ingest_role(&message.role) {
                    continue;
                }
                let tokens = count_tokens(&message.content);
                hot.token_count += tokens;
                hot.messages.push(message);
            }
        }

        loop {
            let should_compact = self
                .hot_buffers
                .get(topic_id)
                .map(|hot| hot.token_count >= self.config.soft_tokens)
                .unwrap_or(false);
            if !should_compact {
                break;
            }
            self.compact_hot_buffer(topic_id)?;
            let hard_ok = self
                .hot_buffers
                .get(topic_id)
                .map(|hot| hot.token_count <= self.config.hard_tokens)
                .unwrap_or(true);
            if hard_ok {
                break;
            }
        }

        Ok(())
    }

    fn compact_hot_buffer(&mut self, topic_id: &str) -> io::Result<()> {
        let hot = self
            .hot_buffers
            .get_mut(topic_id)
            .expect("hot buffer missing");
        if hot.messages.is_empty() {
            return Ok(());
        }
        let split_at = (hot.messages.len() / 2).max(1);
        let to_compact: Vec<Message> = hot.messages.drain(0..split_at).collect();
        let compact_text = messages_to_text(&to_compact);
        let tokens = tokenize_owned(&compact_text);
        let chunk_texts = chunk_tokens(
            &tokens,
            self.config.chunk_target_tokens,
            self.config.chunk_overlap_tokens,
            self.config.chunk_min_tokens,
            self.config.chunk_max_tokens,
        );

        let mut deduped: Vec<(String, Vec<f32>)> = Vec::new();
        for chunk in chunk_texts {
            let embedding = self.embedder.embed(&chunk);
            let mut is_dup = false;
            for (_, existing_emb) in &deduped {
                let sim = cosine_similarity(&embedding, existing_emb);
                if sim >= 0.99 {
                    is_dup = true;
                    break;
                }
            }
            if !is_dup {
                deduped.push((chunk, embedding));
            }
        }

        let created_at = now_ts();
        for (chunk_text, embedding) in deduped {
            let record = ChunkRecord {
                id: Uuid::new_v4(),
                canonical_id: self
                    .topics
                    .get_mut(topic_id)
                    .expect("topic missing")
                    .next_canonical_id(),
                embedding,
                content: chunk_text,
                status: ChunkStatus::Active,
                utility_multiplier: 1.0,
                created_at,
            };
            self.topics
                .get_mut(topic_id)
                .expect("topic missing")
                .append_record(record, &self.config)?;
        }

        hot.token_count = hot.messages.iter().map(|m| count_tokens(&m.content)).sum();
        Ok(())
    }

    fn retrieve(
        &mut self,
        topic_id: &str,
        related_topic_ids: &[String],
        messages: &[Message],
    ) -> io::Result<Vec<ScoredChunk>> {
        let query = extract_query(messages);
        let query_embedding = self.embedder.embed(&query);

        let mut results = Vec::new();
        let mut seen_ids = HashSet::new();

        if let Some(store) = self.topics.get(topic_id) {
            let mut base = store.retrieve(&query_embedding, self.config.top_k)?;
            for chunk in &base {
                seen_ids.insert(chunk.record.id);
            }
            results.append(&mut base);
        }

        for related in related_topic_ids {
            if let Some(store) = self.topics.get(related) {
                let mut rel = store.retrieve(&query_embedding, self.config.top_k)?;
                rel.retain(|chunk| !seen_ids.contains(&chunk.record.id));
                for chunk in &rel {
                    seen_ids.insert(chunk.record.id);
                }
                results.append(&mut rel);
            }
        }

        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(self.config.top_k);
        Ok(results)
    }

    fn apply_corrections(
        &mut self,
        current_topic: &str,
        corrections: &[crate::types::Correction],
    ) -> io::Result<Vec<Signal>> {
        let mut signals = Vec::new();
        let injected_map = self
            .recent_injected
            .get(current_topic)
            .cloned()
            .unwrap_or_default();

        for correction in corrections {
            let mut valid_entries = Vec::new();
            for chunk_id in &correction.chunk_ids {
                let Some(info) = injected_map.get(chunk_id) else {
                    signals.push(Signal {
                        signal_type: SignalType::CorrectionFailed,
                        fill_ratio: None,
                        chunk_id: Some(*chunk_id),
                    });
                    continue;
                };
                let Some(base) = self.get_latest_record(&info.topic_id, chunk_id) else {
                    signals.push(Signal {
                        signal_type: SignalType::CorrectionFailed,
                        fill_ratio: None,
                        chunk_id: Some(*chunk_id),
                    });
                    continue;
                };
                if base.status == ChunkStatus::Deprecated {
                    signals.push(Signal {
                        signal_type: SignalType::CorrectionFailed,
                        fill_ratio: None,
                        chunk_id: Some(*chunk_id),
                    });
                    continue;
                }
                valid_entries.push((info.clone(), base));
            }

            if valid_entries.is_empty() {
                continue;
            }

            match correction.action {
                CorrectiveAction::Helpful | CorrectiveAction::Unhelpful => {
                    for (info, base) in valid_entries {
                        let mut multiplier = base.utility_multiplier;
                        if correction.action == CorrectiveAction::Helpful {
                            multiplier *= self.config.utility_step_ratio;
                            let ceiling = self.config.utility_ceiling();
                            if multiplier > ceiling {
                                multiplier = ceiling;
                            }
                        } else {
                            multiplier /= self.config.utility_step_ratio;
                            let floor = self.config.utility_floor();
                            if multiplier < floor {
                                multiplier = floor;
                            }
                        }
                        let record = ChunkRecord {
                            id: base.id,
                            canonical_id: self
                                .topics
                                .get_mut(current_topic)
                                .expect("topic missing")
                                .next_canonical_id(),
                            embedding: base.embedding.clone(),
                            content: base.content.clone(),
                            status: ChunkStatus::Active,
                            utility_multiplier: multiplier,
                            created_at: now_ts(),
                        };
                        self.topics
                            .get_mut(current_topic)
                            .expect("topic missing")
                            .append_record(record, &self.config)?;
                        let _ = info; // provenance is captured by id reuse
                    }
                }
                CorrectiveAction::Update => {
                    for (_, base) in &valid_entries {
                        let record = ChunkRecord {
                            id: base.id,
                            canonical_id: self
                                .topics
                                .get_mut(current_topic)
                                .expect("topic missing")
                                .next_canonical_id(),
                            embedding: base.embedding.clone(),
                            content: base.content.clone(),
                            status: ChunkStatus::Deprecated,
                            utility_multiplier: base.utility_multiplier,
                            created_at: now_ts(),
                        };
                        self.topics
                            .get_mut(current_topic)
                            .expect("topic missing")
                            .append_record(record, &self.config)?;
                    }
                    if let Some(content) = correction.content.as_ref() {
                        if !content.trim().is_empty() {
                            let embedding = self.embedder.embed(content);
                            let record = ChunkRecord {
                                id: Uuid::new_v4(),
                                canonical_id: self
                                    .topics
                                    .get_mut(current_topic)
                                    .expect("topic missing")
                                    .next_canonical_id(),
                                embedding,
                                content: content.clone(),
                                status: ChunkStatus::Active,
                                utility_multiplier: 1.0,
                                created_at: now_ts(),
                            };
                            self.topics
                                .get_mut(current_topic)
                                .expect("topic missing")
                                .append_record(record, &self.config)?;
                        }
                    }
                }
            }
        }

        Ok(signals)
    }

    fn get_latest_record(&mut self, topic_id: &str, chunk_id: &Uuid) -> Option<ChunkRecord> {
        if !self.topics.contains_key(topic_id) {
            if self.ensure_topic(topic_id).is_err() {
                return None;
            }
        }
        self.topics
            .get(topic_id)
            .and_then(|store| store.latest_entry(chunk_id).cloned())
    }
}

fn insert_before_last_user(messages: &mut Vec<Message>, injection: Message) {
    if let Some(index) = messages.iter().rposition(|m| m.role == "user") {
        messages.insert(index, injection);
    } else {
        messages.push(injection);
    }
}

fn extract_query(messages: &[Message]) -> String {
    messages
        .iter()
        .rev()
        .find(|m| m.role == "user")
        .map(|m| m.content.clone())
        .unwrap_or_default()
}

fn is_ingest_role(role: &str) -> bool {
    matches!(role, "user" | "assistant")
}

fn tokenizer() -> &'static CoreBPE {
    static TOKENIZER: OnceLock<CoreBPE> = OnceLock::new();
    TOKENIZER.get_or_init(|| cl100k_base().expect("failed to load cl100k_base tokenizer"))
}

fn count_tokens(text: &str) -> usize {
    tokenizer().encode_ordinary(text).len()
}

fn tokenize_owned(text: &str) -> Vec<String> {
    match tokenizer().split_by_token_ordinary(text) {
        Ok(tokens) => tokens,
        Err(err) => {
            eprintln!("tokenizer failed; falling back to single chunk: {err}");
            vec![text.to_string()]
        }
    }
}

fn chunk_tokens(
    tokens: &[String],
    target: usize,
    overlap: usize,
    min: usize,
    max: usize,
) -> Vec<String> {
    if tokens.is_empty() {
        return Vec::new();
    }
    let mut chunks = Vec::new();
    let mut start = 0usize;
    while start < tokens.len() {
        let remaining = tokens.len() - start;
        let mut len = if remaining <= max { remaining } else { target.min(max) };
        if len < min && remaining > 0 {
            len = remaining;
        }
        let end = (start + len).min(tokens.len());
        let chunk = tokens[start..end].join("");
        chunks.push(chunk);
        if end == tokens.len() {
            break;
        }
        let step = if len > overlap { len - overlap } else { len };
        start += step.max(1);
    }
    chunks
}

fn messages_to_text(messages: &[Message]) -> String {
    let mut out = String::new();
    for msg in messages {
        out.push_str(&msg.role);
        out.push_str(": ");
        out.push_str(&msg.content);
        out.push('\n');
    }
    out
}

fn longest_common_prefix(a: &[u64], b: &[u64]) -> usize {
    let mut i = 0;
    while i < a.len() && i < b.len() {
        if a[i] != b[i] {
            break;
        }
        i += 1;
    }
    i
}

fn message_digest(message: &Message) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325;
    for b in message.role.as_bytes() {
        hash ^= *b as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash ^= 0u64.wrapping_add(0x9e3779b97f4a7c15);
    hash = hash.wrapping_mul(0x100000001b3);
    for b in message.content.as_bytes() {
        hash ^= *b as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

fn apply_context_budget(
    config: &Config,
    results: &mut Vec<ScoredChunk>,
    current_topic: &str,
) -> (Vec<ScoredChunk>, Option<Signal>, Option<Signal>) {
    let mut total_tokens = 0usize;
    let mut token_counts = Vec::new();
    for chunk in results.iter() {
        let tokens = count_tokens(&chunk.record.content);
        token_counts.push(tokens);
        total_tokens += tokens;
    }

    let mut pressure = None;
    let mut overflow = None;
    let budget = config.context_budget_tokens.max(1);
    let fill_ratio = total_tokens as f32 / budget as f32;

    if fill_ratio > 1.0 {
        overflow = Some(Signal {
            signal_type: SignalType::ContextOverflow,
            fill_ratio: Some(fill_ratio),
            chunk_id: None,
        });
    } else if fill_ratio >= config.context_pressure_ratio {
        pressure = Some(Signal {
            signal_type: SignalType::ContextPressure,
            fill_ratio: Some(fill_ratio),
            chunk_id: None,
        });
    }

    if fill_ratio > 1.0 {
        while total_tokens > budget && !results.is_empty() {
            let idx = results.len() - 1;
            total_tokens -= token_counts[idx];
            token_counts.pop();
            results.pop();
        }
    }

    results.sort_by(|a, b| {
        if a.topic_id == b.topic_id {
            a.record.canonical_id.cmp(&b.record.canonical_id)
        } else if a.topic_id == current_topic {
            std::cmp::Ordering::Less
        } else if b.topic_id == current_topic {
            std::cmp::Ordering::Greater
        } else {
            a.topic_id.cmp(&b.topic_id)
        }
    });

    (results.clone(), pressure, overflow)
}

fn now_ts() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64
}
