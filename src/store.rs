use crate::config::Config;
use crate::types::{ChunkRecord, ChunkStatus, ScoredChunk};
use crate::embed::cosine_similarity;
use hnsw_rs::api::AnnT;
use hnsw_rs::anndists::dist::distances::DistCosine;
use hnsw_rs::hnsw::Hnsw;
use hnsw_rs::hnswio::{HnswIo, ReloadOptions};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fs::{self, File, OpenOptions};
use std::io::{self, BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};
use uuid::Uuid;

type HnswIndex = Hnsw<'static, f32, DistCosine>;

pub struct SegmentInfo {
    #[allow(dead_code)]
    pub id: u32,
    #[allow(dead_code)]
    pub path: PathBuf,
    pub records: Vec<ChunkRecord>,
    pub hnsw: Option<HnswIndex>,
    pub canonical_id_range: (u64, u64),
}

pub struct TopicStore {
    pub topic_id: String,
    #[allow(dead_code)]
    root: PathBuf,
    segments_dir: PathBuf,
    active_path: PathBuf,
    pub latest_by_id: HashMap<Uuid, ChunkRecord>,
    superseded_canonical_ids: HashSet<u64>,
    next_canonical_id: u64,
    active_count: u32,
    active_canonical_min: Option<u64>,
    active_canonical_max: Option<u64>,
    next_segment_id: u32,
    sealed_segments: Vec<SegmentInfo>,
    hnsw_ef_search: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SegmentMeta {
    pub segment_id: u32,
    pub chunk_count: u32,
    pub canonical_id_range: (u64, u64),
    pub sealed_at: i64,
    #[serde(default)]
    pub hnsw_basename: Option<String>,
}

impl TopicStore {
    pub fn load(config: &Config, topic_id: &str) -> io::Result<Self> {
        let root = config.data_dir.join(topic_id);
        let segments_dir = root.join("segments");
        fs::create_dir_all(&segments_dir)?;
        fs::create_dir_all(&root)?;

        let active_path = root.join("active.bin");

        let mut max_canonical = 0u64;

        let mut sealed_segments = Vec::new();
        let mut next_segment_id = 1u32;

        let mut segment_files: Vec<(u32, PathBuf)> = Vec::new();
        if let Ok(entries) = fs::read_dir(&segments_dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if let Some(id) = parse_segment_id(&path) {
                    next_segment_id = next_segment_id.max(id + 1);
                    segment_files.push((id, path));
                }
            }
        }
        segment_files.sort_by_key(|(id, _)| *id);

        for (id, path) in segment_files {
            let records = read_segment_records(&path)?;
            for record in &records {
                if record.canonical_id > max_canonical {
                    max_canonical = record.canonical_id;
                }
            }
            let canonical_id_range = load_segment_range(&segments_dir, id)
                .or_else(|| canonical_range_from_records(&records))
                .unwrap_or((0, 0));
            let hnsw_basename = load_hnsw_basename(&segments_dir, id);
            let hnsw = match hnsw_basename {
                Some(basename) => load_hnsw_index(&segments_dir, &basename),
                None => None,
            };
            sealed_segments.push(SegmentInfo {
                id,
                path,
                records,
                hnsw,
                canonical_id_range,
            });
        }

        let mut active_count = 0u32;
        let mut active_records = Vec::new();
        if active_path.exists() {
            scan_segment(&active_path, |record| {
                active_count += 1;
                if record.canonical_id > max_canonical {
                    max_canonical = record.canonical_id;
                }
                active_records.push(record);
            })?;
        }

        let (active_canonical_min, active_canonical_max) =
            match canonical_range_from_records(&active_records) {
                Some((min, max)) => (Some(min), Some(max)),
                None => (None, None),
            };

        let (latest_by_id, superseded_canonical_ids) =
            rebuild_superseded(&sealed_segments, &active_records);

        Ok(Self {
            topic_id: topic_id.to_string(),
            root,
            segments_dir,
            active_path,
            latest_by_id,
            superseded_canonical_ids,
            next_canonical_id: max_canonical + 1,
            active_count,
            active_canonical_min,
            active_canonical_max,
            next_segment_id,
            sealed_segments,
            hnsw_ef_search: config.hnsw_ef_search,
        })
    }

    pub fn append_record(&mut self, record: ChunkRecord, config: &Config) -> io::Result<()> {
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.active_path)?;
        let line = serde_json::to_string(&record)?;
        writeln!(file, "{line}")?;
        file.flush()?;

        self.active_count += 1;
        self.active_canonical_min = match self.active_canonical_min {
            Some(min) => Some(min.min(record.canonical_id)),
            None => Some(record.canonical_id),
        };
        self.active_canonical_max = match self.active_canonical_max {
            Some(max) => Some(max.max(record.canonical_id)),
            None => Some(record.canonical_id),
        };

        if let Some(existing) = self.latest_by_id.insert(record.id, record.clone()) {
            if existing.canonical_id != record.canonical_id {
                self.superseded_canonical_ids
                    .insert(existing.canonical_id);
            }
        }
        self.maybe_seal(config)?;
        Ok(())
    }

    pub fn next_canonical_id(&mut self) -> u64 {
        let id = self.next_canonical_id;
        self.next_canonical_id += 1;
        id
    }

    pub fn latest_entry(&self, id: &Uuid) -> Option<&ChunkRecord> {
        self.latest_by_id.get(id)
    }

    pub fn retrieve(&self, query_embedding: &[f32], top_k: usize) -> io::Result<Vec<ScoredChunk>> {
        let mut results: Vec<ScoredChunk> = Vec::new();

        if self.active_path.exists() {
            scan_segment(&self.active_path, |record| {
                if !self.is_latest(&record) {
                    return;
                }
                if record.status != ChunkStatus::Active {
                    return;
                }
                let cosine = cosine_similarity(query_embedding, &record.embedding).max(0.0);
                let score = cosine * record.utility_multiplier;
                if score > 0.0 {
                    results.push(ScoredChunk {
                        record,
                        score,
                        topic_id: self.topic_id.clone(),
                    });
                }
            })?;
        }

        for segment in &self.sealed_segments {
            if let Some(hnsw) = &segment.hnsw {
                let ef_search = self.hnsw_ef_search.max(top_k.saturating_mul(2));
                let neighbours = hnsw.search(query_embedding, top_k, ef_search);
                for neigh in neighbours {
                    let idx = neigh.d_id as usize;
                    let Some(record) = segment.records.get(idx) else {
                        continue;
                    };
                    if !self.is_latest(record) {
                        continue;
                    }
                    if record.status != ChunkStatus::Active {
                        continue;
                    }
                    let cosine = (1.0 - neigh.distance).max(0.0);
                    let score = cosine * record.utility_multiplier;
                    if score > 0.0 {
                        results.push(ScoredChunk {
                            record: record.clone(),
                            score,
                            topic_id: self.topic_id.clone(),
                        });
                    }
                }
            } else {
                for record in &segment.records {
                    if !self.is_latest(record) {
                        continue;
                    }
                    if record.status != ChunkStatus::Active {
                        continue;
                    }
                    let cosine = cosine_similarity(query_embedding, &record.embedding).max(0.0);
                    let score = cosine * record.utility_multiplier;
                    if score > 0.0 {
                        results.push(ScoredChunk {
                            record: record.clone(),
                            score,
                            topic_id: self.topic_id.clone(),
                        });
                    }
                }
            }
        }

        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(top_k);
        Ok(results)
    }

    fn is_latest(&self, record: &ChunkRecord) -> bool {
        !self
            .superseded_canonical_ids
            .contains(&record.canonical_id)
    }

    fn maybe_seal(&mut self, config: &Config) -> io::Result<()> {
        if self.active_count < config.active_max_entries {
            return Ok(());
        }
        let Some(min) = self.active_canonical_min else {
            return Ok(());
        };
        let Some(max) = self.active_canonical_max else {
            return Ok(());
        };

        let segment_id = self.next_segment_id;
        let segment_path = self.segments_dir.join(format!("seg_{segment_id:04}.bin"));
        let meta_path = self.segments_dir.join(format!("seg_{segment_id:04}.meta"));

        if self.active_path.exists() {
            fs::rename(&self.active_path, &segment_path)?;
        }

        let records = read_segment_records(&segment_path)?;
        let (hnsw, hnsw_basename) = match build_hnsw_index(
            &records,
            &self.segments_dir,
            segment_id,
            config,
        ) {
            Ok((hnsw, basename)) => (Some(hnsw), Some(basename)),
            Err(err) => {
                eprintln!("failed to build HNSW index for segment {segment_id}: {err}");
                (None, None)
            }
        };

        let meta = SegmentMeta {
            segment_id,
            chunk_count: self.active_count,
            canonical_id_range: (min, max),
            sealed_at: now_ts(),
            hnsw_basename,
        };
        let meta_json = serde_json::to_string_pretty(&meta)?;
        fs::write(meta_path, meta_json)?;

        self.sealed_segments.push(SegmentInfo {
            id: segment_id,
            path: segment_path,
            records,
            hnsw,
            canonical_id_range: (min, max),
        });
        self.next_segment_id += 1;
        self.active_count = 0;
        self.active_canonical_min = None;
        self.active_canonical_max = None;

        let _ = File::create(&self.active_path)?;
        Ok(())
    }
}

fn read_segment_records(path: &Path) -> io::Result<Vec<ChunkRecord>> {
    let mut records = Vec::new();
    scan_segment(path, |record| records.push(record))?;
    Ok(records)
}

fn rebuild_superseded(
    sealed_segments: &[SegmentInfo],
    active_records: &[ChunkRecord],
) -> (HashMap<Uuid, ChunkRecord>, HashSet<u64>) {
    let mut latest_by_id: HashMap<Uuid, ChunkRecord> = HashMap::new();
    let mut superseded: HashSet<u64> = HashSet::new();
    let mut seen_ids: HashSet<Uuid> = HashSet::new();

    for record in active_records.iter().rev() {
        if seen_ids.contains(&record.id) {
            superseded.insert(record.canonical_id);
        } else {
            seen_ids.insert(record.id);
            latest_by_id.insert(record.id, record.clone());
        }
    }

    let mut ordered_segments: Vec<&SegmentInfo> = sealed_segments.iter().collect();
    ordered_segments.sort_by_key(|segment| segment.canonical_id_range.1);
    ordered_segments.reverse();

    for segment in ordered_segments {
        for record in segment.records.iter().rev() {
            if seen_ids.contains(&record.id) {
                superseded.insert(record.canonical_id);
            } else {
                seen_ids.insert(record.id);
                latest_by_id.insert(record.id, record.clone());
            }
        }
    }

    (latest_by_id, superseded)
}

fn build_hnsw_index(
    records: &[ChunkRecord],
    segments_dir: &Path,
    segment_id: u32,
    config: &Config,
) -> io::Result<(HnswIndex, String)> {
    if records.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::Other,
            "empty segment; cannot build HNSW",
        ));
    }
    let max_connections = config.hnsw_max_connections.min(255).max(4);
    let max_elements = records.len().max(1);
    let max_layer = config.hnsw_max_layer.max(1);
    let ef_construction = config.hnsw_ef_construction.max(10);
    let mut hnsw = Hnsw::new(
        max_connections,
        max_elements,
        max_layer,
        ef_construction,
        DistCosine::default(),
    );
    for (idx, record) in records.iter().enumerate() {
        hnsw.insert((record.embedding.as_slice(), idx));
    }
    hnsw.set_searching_mode(true);
    let basename = format!("seg_{segment_id:04}");
    let used_basename = hnsw
        .file_dump(segments_dir, &basename)
        .map_err(|err| io::Error::new(io::ErrorKind::Other, err.to_string()))?;
    Ok((hnsw, used_basename))
}

fn load_hnsw_basename(segments_dir: &Path, segment_id: u32) -> Option<String> {
    let meta_path = segments_dir.join(format!("seg_{segment_id:04}.meta"));
    if let Ok(meta) = fs::read_to_string(meta_path) {
        if let Ok(meta) = serde_json::from_str::<SegmentMeta>(&meta) {
            if let Some(name) = meta.hnsw_basename {
                return Some(name);
            }
        }
    }
    Some(format!("seg_{segment_id:04}"))
}

fn load_segment_range(segments_dir: &Path, segment_id: u32) -> Option<(u64, u64)> {
    let meta_path = segments_dir.join(format!("seg_{segment_id:04}.meta"));
    let meta = std::fs::read_to_string(meta_path).ok()?;
    let meta: SegmentMeta = serde_json::from_str(&meta).ok()?;
    Some(meta.canonical_id_range)
}

fn load_hnsw_index(segments_dir: &Path, basename: &str) -> Option<HnswIndex> {
    let graph_path = segments_dir.join(format!("{basename}.hnsw.graph"));
    let data_path = segments_dir.join(format!("{basename}.hnsw.data"));
    if !graph_path.exists() || !data_path.exists() {
        return None;
    }
    let mut hnsw_io = HnswIo::new_with_options(segments_dir, basename, ReloadOptions::new(false));
    let loaded = hnsw_io.load_hnsw::<f32, DistCosine>().ok()?;
    // Safety: we disable mmap, so loaded data is owned by the HNSW structure.
    let mut hnsw: HnswIndex = unsafe { std::mem::transmute(loaded) };
    hnsw.set_searching_mode(true);
    Some(hnsw)
}

fn scan_segment<F>(path: &Path, mut f: F) -> io::Result<()>
where
    F: FnMut(ChunkRecord),
{
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    for line in reader.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        match serde_json::from_str::<ChunkRecord>(&line) {
            Ok(record) => f(record),
            Err(err) => {
                eprintln!("failed to parse chunk record in {:?}: {err}", path);
                continue;
            }
        }
    }
    Ok(())
}

fn parse_segment_id(path: &Path) -> Option<u32> {
    let file_name = path.file_name()?.to_string_lossy();
    if !file_name.starts_with("seg_") || !file_name.ends_with(".bin") {
        return None;
    }
    let id_part = file_name.trim_start_matches("seg_").trim_end_matches(".bin");
    id_part.parse::<u32>().ok()
}

fn canonical_range_from_records(records: &[ChunkRecord]) -> Option<(u64, u64)> {
    if records.is_empty() {
        return None;
    }
    let mut min = None;
    let mut max = None;
    for record in records {
        min = Some(min.map_or(record.canonical_id, |v: u64| v.min(record.canonical_id)));
        max = Some(max.map_or(record.canonical_id, |v: u64| v.max(record.canonical_id)));
    }
    Some((min.unwrap_or(0), max.unwrap_or(0)))
}

fn now_ts() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64
}
