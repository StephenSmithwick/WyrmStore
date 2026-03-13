use memoryd::embed::EmbedderApi;

#[derive(Debug)]
pub struct HashEmbedder {
    dim: usize,
}

impl HashEmbedder {
    pub fn new(dim: usize) -> Self {
        Self { dim }
    }
}

impl EmbedderApi for HashEmbedder {
    fn embed(&self, text: &str) -> Vec<f32> {
        let mut vec = vec![0.0f32; self.dim];
        for token in text.split_whitespace() {
            let token = token.to_ascii_lowercase();
            let hash = fnv1a64(token.as_bytes());
            let idx = (hash % self.dim as u64) as usize;
            vec[idx] += 1.0;
        }
        l2_normalize(vec)
    }
}

fn l2_normalize(mut vec: Vec<f32>) -> Vec<f32> {
    let mut norm = 0.0f32;
    for v in &vec {
        norm += v * v;
    }
    if norm <= f32::EPSILON {
        return vec;
    }
    let inv = 1.0 / norm.sqrt();
    for v in &mut vec {
        *v *= inv;
    }
    vec
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325;
    for b in bytes {
        hash ^= *b as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}
