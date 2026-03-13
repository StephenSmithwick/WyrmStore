use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use uuid::Uuid;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Message {
    pub role: String,
    pub content: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<serde_json::Value>,
    #[serde(flatten)]
    pub extra: BTreeMap<String, serde_json::Value>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DaemonRequest {
    pub messages: Vec<Message>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub memory_in: Option<MemoryIn>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DaemonResponse {
    pub messages: Vec<Message>,
    pub memory_out: MemoryOut,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MemoryIn {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub topic_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub related_topic_ids: Option<Vec<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub corrections: Option<Vec<Correction>>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Correction {
    pub chunk_ids: Vec<Uuid>,
    pub action: CorrectiveAction,
    pub reason: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "PascalCase")]
pub enum CorrectiveAction {
    Update,
    Helpful,
    Unhelpful,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MemoryOut {
    pub injected_chunks: Vec<InjectedChunk>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub signals: Option<Vec<Signal>>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct InjectedChunk {
    pub id: Uuid,
    pub topic_id: String,
    pub canonical_id: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SignalType {
    ContextPressure,
    ContextOverflow,
    CorrectionFailed,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Signal {
    #[serde(rename = "type")]
    pub signal_type: SignalType,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fill_ratio: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chunk_id: Option<Uuid>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ChunkStatus {
    Active,
    Deprecated,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChunkRecord {
    pub id: Uuid,
    pub canonical_id: u64,
    pub embedding: Vec<f32>,
    pub content: String,
    pub status: ChunkStatus,
    pub utility_multiplier: f32,
    pub created_at: i64,
}

#[derive(Clone, Debug)]
pub struct ScoredChunk {
    pub record: ChunkRecord,
    pub score: f32,
    pub topic_id: String,
}

#[derive(Clone, Debug)]
pub struct InjectedChunkInfo {
    pub topic_id: String,
    #[allow(dead_code)]
    pub canonical_id: u64,
}
