use serde::Deserialize;
use std::env;
use std::io;
use std::path::{Path, PathBuf};

#[derive(Clone, Debug, Deserialize)]
#[serde(default)]
pub struct Config {
    pub data_dir: PathBuf,
    pub socket_path: Option<PathBuf>,
    pub soft_tokens: usize,
    pub hard_tokens: usize,
    pub chunk_min_tokens: usize,
    pub chunk_max_tokens: usize,
    pub chunk_target_tokens: usize,
    pub chunk_overlap_tokens: usize,
    pub active_max_entries: u32,
    pub top_k: usize,
    #[allow(dead_code)]
    pub fastembed_show_progress: bool,
    pub hnsw_max_connections: usize,
    pub hnsw_ef_construction: usize,
    pub hnsw_ef_search: usize,
    pub hnsw_max_layer: usize,
    pub context_budget_tokens: usize,
    pub context_pressure_ratio: f32,
    pub utility_step_ratio: f32,
    pub max_helpful_steps: i32,
    pub max_unhelpful_steps: i32,
    pub framing: bool,
}

impl Config {
    pub fn from_env() -> io::Result<Self> {
        let mut config = if let Ok(path) = env::var("MEMD_CONFIG_PATH") {
            Self::from_toml_path(path)?
        } else {
            Self::default()
        };
        config.apply_env_overrides();
        Ok(config)
    }

    pub fn from_toml_path<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        let contents = std::fs::read_to_string(path)?;
        let config = toml::from_str(&contents)
            .map_err(|err| io::Error::new(io::ErrorKind::InvalidData, err))?;
        Ok(config)
    }

    pub fn apply_env_overrides(&mut self) {
        if let Ok(data_dir) = env::var("MEMD_DATA_DIR") {
            self.data_dir = PathBuf::from(data_dir);
        }
        if let Ok(socket_path) = env::var("MEMD_SOCKET_PATH") {
            self.socket_path = Some(PathBuf::from(socket_path));
        }
        if let Some(value) = env_opt_usize("MEMD_SOFT_TOKENS") {
            self.soft_tokens = value;
        }
        if let Some(value) = env_opt_usize("MEMD_HARD_TOKENS") {
            self.hard_tokens = value;
        }
        if let Some(value) = env_opt_usize("MEMD_CHUNK_MIN_TOKENS") {
            self.chunk_min_tokens = value;
        }
        if let Some(value) = env_opt_usize("MEMD_CHUNK_MAX_TOKENS") {
            self.chunk_max_tokens = value;
        }
        if let Some(value) = env_opt_usize("MEMD_CHUNK_TARGET_TOKENS") {
            self.chunk_target_tokens = value;
        }
        if let Some(value) = env_opt_usize("MEMD_CHUNK_OVERLAP_TOKENS") {
            self.chunk_overlap_tokens = value;
        }
        if let Some(value) = env_opt_u32("MEMD_ACTIVE_MAX_ENTRIES") {
            self.active_max_entries = value;
        }
        if let Some(value) = env_opt_usize("MEMD_TOP_K") {
            self.top_k = value;
        }
        if let Some(value) = env_opt_bool("MEMD_FASTEMBED_SHOW_PROGRESS") {
            self.fastembed_show_progress = value;
        }
        if let Some(value) = env_opt_usize("MEMD_HNSW_M") {
            self.hnsw_max_connections = value;
        }
        if let Some(value) = env_opt_usize("MEMD_HNSW_EF_CONSTRUCTION") {
            self.hnsw_ef_construction = value;
        }
        if let Some(value) = env_opt_usize("MEMD_HNSW_EF_SEARCH") {
            self.hnsw_ef_search = value;
        }
        if let Some(value) = env_opt_usize("MEMD_HNSW_MAX_LAYER") {
            self.hnsw_max_layer = value;
        }
        if let Some(value) = env_opt_usize("MEMD_CONTEXT_BUDGET_TOKENS") {
            self.context_budget_tokens = value;
        }
        if let Some(value) = env_opt_f32("MEMD_CONTEXT_PRESSURE_RATIO") {
            self.context_pressure_ratio = value;
        }
        if let Some(value) = env_opt_f32("MEMD_UTILITY_STEP_RATIO") {
            self.utility_step_ratio = value;
        }
        if let Some(value) = env_opt_i32("MEMD_MAX_HELPFUL_STEPS") {
            self.max_helpful_steps = value;
        }
        if let Some(value) = env_opt_i32("MEMD_MAX_UNHELPFUL_STEPS") {
            self.max_unhelpful_steps = value;
        }
        if let Some(value) = env_opt_bool("MEMD_FRAMING") {
            self.framing = value;
        }
    }

    pub fn utility_ceiling(&self) -> f32 {
        self.utility_step_ratio.powi(self.max_helpful_steps)
    }

    pub fn utility_floor(&self) -> f32 {
        self.utility_step_ratio.powi(-self.max_unhelpful_steps)
    }
}

fn env_opt_usize(key: &str) -> Option<usize> {
    env::var(key).ok().and_then(|v| v.parse().ok())
}

fn env_opt_u32(key: &str) -> Option<u32> {
    env::var(key).ok().and_then(|v| v.parse().ok())
}

fn env_opt_i32(key: &str) -> Option<i32> {
    env::var(key).ok().and_then(|v| v.parse().ok())
}

fn env_opt_f32(key: &str) -> Option<f32> {
    env::var(key).ok().and_then(|v| v.parse().ok())
}

fn env_opt_bool(key: &str) -> Option<bool> {
    env::var(key).ok().map(|v| matches!(v.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
}

impl Default for Config {
    fn default() -> Self {
        Self {
            data_dir: PathBuf::from("memory"),
            socket_path: None,
            soft_tokens: 3500,
            hard_tokens: 4000,
            chunk_min_tokens: 50,
            chunk_max_tokens: 300,
            chunk_target_tokens: 180,
            chunk_overlap_tokens: 20,
            active_max_entries: 5000,
            top_k: 8,
            fastembed_show_progress: false,
            hnsw_max_connections: 16,
            hnsw_ef_construction: 200,
            hnsw_ef_search: 50,
            hnsw_max_layer: 16,
            context_budget_tokens: 2000,
            context_pressure_ratio: 0.8,
            utility_step_ratio: 1.5,
            max_helpful_steps: 10,
            max_unhelpful_steps: 4,
            framing: false,
        }
    }
}
