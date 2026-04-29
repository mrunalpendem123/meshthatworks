//! Parse a safetensors file's header and return per-expert byte ranges.
//!
//! For an MoE model like Qwen3-30B-A3B or OLMoE, expert weights live in
//! tensors named `model.layers.{L}.mlp.experts.{E}.*` (with `gate_proj`,
//! `up_proj`, `down_proj` × `weight` | `scales` | `biases` for 4-bit quants).
//! `parse_expert_layout` walks the safetensors index, groups tensors by
//! `(layer, expert)`, and returns the absolute byte offsets — which feeds
//! `MemoryAdvisor::advise` so the cache can issue per-expert madvise.
//!
//! Importantly: in mlx-community 4-bit quants, an expert's tensors are
//! *not* contiguous. `down_proj.weight` may be at 3.3 GB while
//! `up_proj.scales` is at 845 MB. So `ExpertLayout::ranges` is a `Vec`,
//! not a single (offset, len). The advisor handles each range independently.
//!
//! Format reference: https://github.com/huggingface/safetensors
//!   - 8 bytes LE u64 header_len
//!   - header_len bytes UTF-8 JSON: { name: { dtype, shape, data_offsets: [start,end] }, "__metadata__": {...} }
//!   - rest: raw tensor bytes; data_offsets are relative to the start of
//!     the data section (i.e. add 8 + header_len for absolute).

use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

use anyhow::{Context, bail};
use serde::Deserialize;

use crate::ExpertId;

/// Byte ranges (absolute file offsets) for one expert's tensors.
#[derive(Debug, Clone, Default)]
pub struct ExpertLayout {
    /// `(absolute_offset_in_file, len_in_bytes)`.
    pub ranges: Vec<(u64, u64)>,
}

impl ExpertLayout {
    pub fn total_bytes(&self) -> u64 {
        self.ranges.iter().map(|(_, l)| *l).sum()
    }
}

#[derive(Deserialize)]
struct TensorEntry {
    #[allow(dead_code)]
    dtype: String,
    #[allow(dead_code)]
    shape: Vec<i64>,
    data_offsets: [u64; 2],
}

/// Walk the safetensors header at `path` and return one `ExpertLayout` per
/// `(layer, expert)` referenced by any tensor name matching
/// `model.layers.{L}.mlp.experts.{E}.*`. Tensors not matching that pattern
/// (embedding, attention, lm_head, dense layers) are silently ignored —
/// the cache only manages experts.
pub fn parse_expert_layout(
    path: impl AsRef<Path>,
) -> anyhow::Result<HashMap<ExpertId, ExpertLayout>> {
    let path = path.as_ref();
    let mut f = File::open(path)
        .with_context(|| format!("open {}", path.display()))?;

    let mut len_buf = [0u8; 8];
    f.read_exact(&mut len_buf).context("read header length")?;
    let header_len = u64::from_le_bytes(len_buf);
    if header_len > 100 * 1024 * 1024 {
        bail!("safetensors header_len {} looks too large", header_len);
    }
    let mut header = vec![0u8; header_len as usize];
    f.read_exact(&mut header).context("read header bytes")?;

    let header_json: HashMap<String, serde_json::Value> =
        serde_json::from_slice(&header).context("decode safetensors header JSON")?;

    let data_section_start: u64 = 8 + header_len;

    // sanity check that we read past the header
    f.seek(SeekFrom::Start(data_section_start)).ok();

    let mut out: HashMap<ExpertId, ExpertLayout> = HashMap::new();
    for (name, value) in header_json {
        if name == "__metadata__" {
            continue;
        }
        let Some((layer, expert)) = parse_expert_name(&name) else {
            continue;
        };
        let entry: TensorEntry = serde_json::from_value(value)
            .with_context(|| format!("decode tensor entry for {name}"))?;
        let [start, end] = entry.data_offsets;
        if end < start {
            bail!("tensor {name} has end < start");
        }
        let abs_offset = data_section_start + start;
        let len = end - start;
        out.entry(ExpertId::new(layer, expert))
            .or_default()
            .ranges
            .push((abs_offset, len));
    }

    // Sort each expert's ranges by offset so madvise calls are roughly
    // sequential when the caller iterates them in order. Doesn't matter
    // for correctness, just makes I/O patterns slightly nicer.
    for layout in out.values_mut() {
        layout.ranges.sort_by_key(|(o, _)| *o);
    }

    Ok(out)
}

/// Parse a tensor name like `model.layers.12.mlp.experts.37.up_proj.scales`
/// into `(12, 37)`. Returns `None` for non-expert tensors.
fn parse_expert_name(name: &str) -> Option<(u32, u32)> {
    // Split on '.' and look for "layers" then a number, then "experts" then a number.
    let parts: Vec<&str> = name.split('.').collect();
    let mut layer = None;
    let mut expert = None;
    for i in 0..parts.len() {
        if parts[i] == "layers" && i + 1 < parts.len() {
            if let Ok(n) = parts[i + 1].parse::<u32>() {
                layer = Some(n);
            }
        }
        if parts[i] == "experts" && i + 1 < parts.len() {
            if let Ok(n) = parts[i + 1].parse::<u32>() {
                expert = Some(n);
            }
        }
    }
    match (layer, expert) {
        (Some(l), Some(e)) => Some((l, e)),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn parses_expert_names() {
        assert_eq!(
            parse_expert_name("model.layers.0.mlp.experts.0.gate_proj.weight"),
            Some((0, 0))
        );
        assert_eq!(
            parse_expert_name("model.layers.12.mlp.experts.37.up_proj.scales"),
            Some((12, 37))
        );
        // Non-expert tensors return None.
        assert_eq!(parse_expert_name("model.embed_tokens.weight"), None);
        assert_eq!(
            parse_expert_name("model.layers.0.self_attn.q_proj.weight"),
            None
        );
        assert_eq!(parse_expert_name("lm_head.weight"), None);
    }

    /// Build a tiny synthetic safetensors file so the parser is testable
    /// without a real model on disk.
    fn synth_file(path: &Path, header: &str, data: &[u8]) {
        let mut f = std::fs::File::create(path).expect("create");
        let header_bytes = header.as_bytes();
        f.write_all(&(header_bytes.len() as u64).to_le_bytes())
            .unwrap();
        f.write_all(header_bytes).unwrap();
        f.write_all(data).unwrap();
        f.sync_all().unwrap();
    }

    #[test]
    fn parses_synthetic_two_expert_file() {
        let dir = std::env::temp_dir();
        let path = dir.join(format!("mtw-st-{}.safetensors", std::process::id()));

        let header = serde_json::json!({
            "__metadata__": {"format": "pt"},
            "model.embed_tokens.weight": {
                "dtype": "F32", "shape": [4, 4], "data_offsets": [0, 64]
            },
            "model.layers.0.mlp.experts.0.gate_proj.weight": {
                "dtype": "I8", "shape": [4, 4], "data_offsets": [64, 80]
            },
            "model.layers.0.mlp.experts.0.up_proj.weight": {
                "dtype": "I8", "shape": [4, 4], "data_offsets": [80, 96]
            },
            "model.layers.0.mlp.experts.1.gate_proj.weight": {
                "dtype": "I8", "shape": [4, 4], "data_offsets": [96, 112]
            },
        })
        .to_string();

        synth_file(&path, &header, &vec![0u8; 112]);

        let layout = parse_expert_layout(&path).expect("parse");
        assert_eq!(layout.len(), 2);

        let e00 = layout.get(&ExpertId::new(0, 0)).expect("0,0 present");
        assert_eq!(e00.ranges.len(), 2);
        assert_eq!(e00.total_bytes(), 32);

        let e01 = layout.get(&ExpertId::new(0, 1)).expect("0,1 present");
        assert_eq!(e01.ranges.len(), 1);
        assert_eq!(e01.total_bytes(), 16);

        // Embed and lm_head are not experts — should be filtered out.
        assert!(!layout.contains_key(&ExpertId::new(0, 99)));

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn ranges_are_sorted_by_offset() {
        let dir = std::env::temp_dir();
        let path = dir.join(format!("mtw-st-sort-{}.safetensors", std::process::id()));
        // Same expert, ranges given out of order in JSON.
        let header = serde_json::json!({
            "model.layers.0.mlp.experts.0.up_proj.weight": {
                "dtype": "I8", "shape": [4, 4], "data_offsets": [200, 216]
            },
            "model.layers.0.mlp.experts.0.gate_proj.weight": {
                "dtype": "I8", "shape": [4, 4], "data_offsets": [0, 16]
            },
            "model.layers.0.mlp.experts.0.down_proj.weight": {
                "dtype": "I8", "shape": [4, 4], "data_offsets": [100, 116]
            },
        })
        .to_string();
        synth_file(&path, &header, &vec![0u8; 216]);
        let layout = parse_expert_layout(&path).expect("parse");
        let ranges = &layout.get(&ExpertId::new(0, 0)).unwrap().ranges;
        assert_eq!(ranges[0].0 < ranges[1].0, true);
        assert_eq!(ranges[1].0 < ranges[2].0, true);
        std::fs::remove_file(&path).ok();
    }
}
