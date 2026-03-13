use crate::config::Config;
use std::io;

pub trait EmbedderApi: Send + Sync {
    fn embed(&self, text: &str) -> Vec<f32>;
}

pub fn build_embedder(config: &Config) -> io::Result<Box<dyn EmbedderApi>> {
    {
        let fast = FastEmbedder::new(config)?;
        Ok(Box::new(fast))
    }
}

struct FastEmbedder {
    model: std::sync::Mutex<fastembed::TextEmbedding>,
}

impl FastEmbedder {
    fn new(config: &Config) -> io::Result<Self> {
        use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};
        let options = InitOptions::new(EmbeddingModel::AllMiniLML6V2)
            .with_show_download_progress(config.fastembed_show_progress);
        let model = TextEmbedding::try_new(options).map_err(|err| {
            io::Error::new(
                io::ErrorKind::Other,
                format!("fastembed init failed: {err}"),
            )
        })?;
        Ok(Self {
            model: std::sync::Mutex::new(model),
        })
    }
}

impl EmbedderApi for FastEmbedder {
    fn embed(&self, text: &str) -> Vec<f32> {
        let mut model = match self.model.lock() {
            Ok(model) => model,
            Err(_) => {
                eprintln!("fastembed embed failed: model lock poisoned");
                return Vec::new();
            }
        };
        match model.embed(vec![text], None) {
            Ok(mut embeddings) => embeddings.pop().unwrap_or_default(),
            Err(err) => {
                eprintln!("fastembed embed failed: {err}");
                Vec::new()
            }
        }
    }
}

pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let mut dot = 0.0f32;
    for i in 0..a.len() {
        dot += a[i] * b[i];
    }
    dot
}
