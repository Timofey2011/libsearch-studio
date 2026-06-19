//! Local embedding for the LibSearch Studio engine.
//!
//! Runs the multilingual `bge-m3` model (exported to ONNX by `scripts/export_onnx.py`)
//! via ONNX Runtime (`ort`) + the HF `tokenizers` crate, matching the Python engine:
//! 1024-d dense vectors, **CLS pooling** (`last_hidden_state[:, 0, :]`), **L2-normalized**.
//!
//! The embedding model must match the one used at index time; the parity test
//! (`tests/parity.rs`, `--features models`) proves equivalence to the Python
//! `sentence-transformers` bge-m3 (cosine ≥ 0.999).

use std::path::Path;

use ndarray::Array2;
use ort::session::Session;
use tokenizers::Tokenizer;

pub const BGE_M3_DIM: usize = 1024;
const MAX_LEN: usize = 512;

#[derive(Debug, thiserror::Error)]
pub enum EmbedError {
    #[error("onnx runtime: {0}")]
    Ort(#[from] ort::Error),
    #[error("tokenizer: {0}")]
    Tokenizer(String),
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error("model output missing or malformed: {0}")]
    Output(String),
}

/// Cosine similarity of two equal-length vectors. Returns 0.0 if either is zero.
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        dot / (na * nb)
    }
}

fn l2_normalize(v: &mut [f32]) {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

/// Dense text embedder backed by an ONNX `bge-m3` model directory
/// (`model.onnx` + `tokenizer.json`).
pub struct Embedder {
    session: Session,
    tokenizer: Tokenizer,
    dim: usize,
}

impl Embedder {
    /// Load a model directory containing `model.onnx` and `tokenizer.json`.
    pub fn load(model_dir: impl AsRef<Path>) -> Result<Self, EmbedError> {
        let dir = model_dir.as_ref();
        let mut tokenizer = Tokenizer::from_file(dir.join("tokenizer.json"))
            .map_err(|e| EmbedError::Tokenizer(e.to_string()))?;
        tokenizer
            .with_padding(Some(tokenizers::PaddingParams {
                strategy: tokenizers::PaddingStrategy::BatchLongest,
                ..Default::default()
            }))
            .with_truncation(Some(tokenizers::TruncationParams {
                max_length: MAX_LEN,
                ..Default::default()
            }))
            .map_err(|e| EmbedError::Tokenizer(e.to_string()))?;

        let session = Session::builder()?.commit_from_file(dir.join("model.onnx"))?;
        Ok(Self {
            session,
            tokenizer,
            dim: BGE_M3_DIM,
        })
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Embed passages as-is (bge-m3 needs no query-instruction prefix). Normalized.
    pub fn embed(&mut self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbedError> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let encodings = self
            .tokenizer
            .encode_batch(texts.to_vec(), true)
            .map_err(|e| EmbedError::Tokenizer(e.to_string()))?;

        let batch = encodings.len();
        let seq = encodings[0].get_ids().len();
        let mut ids = Array2::<i64>::zeros((batch, seq));
        let mut mask = Array2::<i64>::zeros((batch, seq));
        for (i, enc) in encodings.iter().enumerate() {
            for (j, &id) in enc.get_ids().iter().enumerate() {
                ids[[i, j]] = id as i64;
            }
            for (j, &m) in enc.get_attention_mask().iter().enumerate() {
                mask[[i, j]] = m as i64;
            }
        }

        let outputs = self
            .session
            .run(ort::inputs!["input_ids" => ids, "attention_mask" => mask]?)?;

        // last_hidden_state: [batch, seq, hidden]. Dense embedding = CLS row, L2-normed.
        let (shape, data) = outputs["last_hidden_state"]
            .try_extract_raw_tensor::<f32>()
            .map_err(|e| EmbedError::Output(e.to_string()))?;
        if shape.len() != 3 {
            return Err(EmbedError::Output(format!(
                "expected rank-3 output, got {shape:?}"
            )));
        }
        let hidden = shape[2] as usize;

        let mut out = Vec::with_capacity(batch);
        for i in 0..batch {
            // CLS is token 0: offset i*seq*hidden + 0*hidden.
            let start = i * seq * hidden;
            let mut v = data[start..start + hidden].to_vec();
            l2_normalize(&mut v);
            out.push(v);
        }
        Ok(out)
    }

    /// Embed a single query (same representation as passages for bge-m3).
    pub fn embed_query(&mut self, text: &str) -> Result<Vec<f32>, EmbedError> {
        Ok(self.embed(&[text])?.remove(0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cosine_of_identical_is_one() {
        let v = vec![0.1, 0.2, 0.3, 0.4];
        assert!((cosine(&v, &v) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_of_orthogonal_is_zero() {
        assert!(cosine(&[1.0, 0.0], &[0.0, 1.0]).abs() < 1e-6);
    }

    #[test]
    fn l2_normalize_makes_unit_vector() {
        let mut v = vec![3.0, 4.0];
        l2_normalize(&mut v);
        let norm = (v[0] * v[0] + v[1] * v[1]).sqrt();
        assert!((norm - 1.0).abs() < 1e-6);
    }
}
