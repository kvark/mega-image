use std::error::Error;
use std::path::Path;

use meganeura::graph::{DType, NodeId};
use meganeura::load::gguf::{GgmlType, GgufModel, GgufTensor};
use meganeura::{Graph, Session};
use tokenizers::Tokenizer;

pub const HIDDEN: usize = 4096;
const LAYERS: usize = 36;
const HEADS: u32 = 32;
const KV_HEADS: u32 = 8;
const HEAD_DIM: u32 = 128;
const KV_DIM: usize = KV_HEADS as usize * HEAD_DIM as usize;
const FFN: usize = 12_288;
const EPS: f32 = 1.0e-6;
const THETA: f32 = 5_000_000.0;
const SYSTEM: &str = "Comprehend and analyze the provided prompt.";

#[derive(Clone, Copy)]
enum MatrixKind {
    Dense,
    Packed(DType),
}

pub struct PromptTokens {
    pub ids: Vec<u32>,
    pub drop: usize,
}

pub fn tokenize(tokenizer_path: &Path, prompt: &str) -> Result<PromptTokens, Box<dyn Error>> {
    let tokenizer = Tokenizer::from_file(tokenizer_path)
        .map_err(|error| format!("loading tokenizer: {error}"))?;
    tokenize_with(&tokenizer, prompt)
}

fn tokenize_with(tokenizer: &Tokenizer, prompt: &str) -> Result<PromptTokens, Box<dyn Error>> {
    let prompt = if prompt.is_empty() { " " } else { prompt };
    let system = format!("<|im_start|>system\n{SYSTEM}<|im_end|>\n");
    let full = format!("{system}<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n");
    let drop = tokenizer
        .encode(system, false)
        .map_err(|error| format!("tokenizing system prompt: {error}"))?
        .len();
    let ids = tokenizer
        .encode(full, false)
        .map_err(|error| format!("tokenizing prompt: {error}"))?
        .get_ids()
        .to_vec();
    if drop >= ids.len() {
        return Err("prompt template produced no user/assistant tokens".into());
    }
    Ok(PromptTokens { ids, drop })
}

fn matrix_kind(tensor: &GgufTensor) -> Result<MatrixKind, Box<dyn Error>> {
    Ok(match tensor.ggml_type {
        GgmlType::F32 | GgmlType::F16 | GgmlType::BF16 => MatrixKind::Dense,
        _ => MatrixKind::Packed(tensor.packed_dtype()?),
    })
}

fn declare_matrix(
    graph: &mut Graph,
    model: &GgufModel,
    name: &str,
    input: usize,
    output: usize,
) -> Result<(NodeId, MatrixKind), Box<dyn Error>> {
    let tensor = model
        .tensors
        .get(name)
        .ok_or_else(|| format!("text GGUF has no tensor `{name}`"))?;
    if tensor.dims.as_slice() != [input, output] {
        return Err(format!(
            "`{name}` has dimensions {:?}, expected [{input}, {output}]",
            tensor.dims
        )
        .into());
    }
    let kind = matrix_kind(tensor)?;
    let node = match kind {
        MatrixKind::Dense => graph.parameter(name, &[output, input]),
        MatrixKind::Packed(DType::Q40) => graph.parameter_q40(name, &[input, output]),
        MatrixKind::Packed(DType::Q4_0) => graph.parameter_q4(name, &[input, output]),
        MatrixKind::Packed(DType::Q8_0) => graph.parameter_q8(name, &[input, output]),
        MatrixKind::Packed(DType::Q4K) => graph.parameter_q4k(name, &[input, output]),
        MatrixKind::Packed(DType::Q5K) => graph.parameter_q5k(name, &[input, output]),
        MatrixKind::Packed(DType::Q6K) => graph.parameter_q6k(name, &[input, output]),
        MatrixKind::Packed(DType::Q3K) => graph.parameter_q3k(name, &[input, output]),
        MatrixKind::Packed(other) => {
            return Err(format!("unsupported text matrix dtype {other:?}").into());
        }
    };
    Ok((node, kind))
}

fn project(graph: &mut Graph, x: NodeId, matrix: (NodeId, MatrixKind)) -> NodeId {
    match matrix.1 {
        MatrixKind::Dense => graph.matmul_bt(x, matrix.0),
        MatrixKind::Packed(_) => graph.matmul(x, matrix.0),
    }
}

fn projection(
    graph: &mut Graph,
    model: &GgufModel,
    name: &str,
    x: NodeId,
    input: usize,
    output: usize,
) -> Result<NodeId, Box<dyn Error>> {
    let matrix = declare_matrix(graph, model, name, input, output)?;
    Ok(project(graph, x, matrix))
}

fn norm(
    graph: &mut Graph,
    model: &GgufModel,
    name: &str,
    width: usize,
) -> Result<NodeId, Box<dyn Error>> {
    let tensor = model
        .tensors
        .get(name)
        .ok_or_else(|| format!("text GGUF has no tensor `{name}`"))?;
    if tensor.dims.as_slice() != [width] {
        return Err(format!(
            "`{name}` has dimensions {:?}, expected [{width}]",
            tensor.dims
        )
        .into());
    }
    Ok(graph.parameter(name, &[width]))
}

/// Decode only the prompt's rows from the quantized token table.
pub fn token_embeddings(model: &GgufModel, ids: &[u32]) -> Result<Vec<f32>, Box<dyn Error>> {
    let table = model
        .tensors
        .get("token_embd.weight")
        .ok_or("text GGUF has no `token_embd.weight`")?;
    if table.dims.first() != Some(&HIDDEN) {
        return Err(format!("token embedding has unexpected shape {:?}", table.dims).into());
    }
    let mut output = Vec::with_capacity(ids.len() * HIDDEN);
    for &id in ids {
        output.extend(table.column_f32(id as usize)?);
    }
    Ok(output)
}

/// Build Qwen3-VL's text-only language-model path.
///
/// The output deliberately bypasses the model's final RMSNorm. Qwen-Image
/// 2.1 was trained on the last decoder block output before that norm.
pub fn build(
    graph: &mut Graph,
    model: &GgufModel,
    sequence: usize,
    drop: usize,
) -> Result<NodeId, Box<dyn Error>> {
    if model.architecture() != Some("qwen3vl") {
        return Err(format!(
            "expected qwen3vl text GGUF, found {:?}",
            model.architecture()
        )
        .into());
    }
    if drop >= sequence {
        return Err("text system-token drop must be shorter than the sequence".into());
    }
    let mut x = graph.input("token_embeddings", &[sequence, HIDDEN]);

    for layer in 0..LAYERS {
        let prefix = format!("blk.{layer}");
        let input_norm = norm(graph, model, &format!("{prefix}.attn_norm.weight"), HIDDEN)?;
        let h = graph.rms_norm(x, input_norm, EPS);

        let q = projection(
            graph,
            model,
            &format!("{prefix}.attn_q.weight"),
            h,
            HIDDEN,
            HIDDEN,
        )?;
        let k = projection(
            graph,
            model,
            &format!("{prefix}.attn_k.weight"),
            h,
            HIDDEN,
            KV_DIM,
        )?;
        let v = projection(
            graph,
            model,
            &format!("{prefix}.attn_v.weight"),
            h,
            HIDDEN,
            KV_DIM,
        )?;
        let q_norm = norm(
            graph,
            model,
            &format!("{prefix}.attn_q_norm.weight"),
            HEAD_DIM as usize,
        )?;
        let k_norm = norm(
            graph,
            model,
            &format!("{prefix}.attn_k_norm.weight"),
            HEAD_DIM as usize,
        )?;
        let q = graph.reshape(q, &[sequence * HEADS as usize, HEAD_DIM as usize]);
        let q = graph.rms_norm(q, q_norm, EPS);
        let q = graph.reshape(q, &[sequence, HIDDEN]);
        let k = graph.reshape(k, &[sequence * KV_HEADS as usize, HEAD_DIM as usize]);
        let k = graph.rms_norm(k, k_norm, EPS);
        let k = graph.reshape(k, &[sequence, KV_DIM]);
        // Text positions are identical on all three Qwen3-VL M-RoPE axes,
        // which reduces exactly to ordinary half-split RoPE.
        let q = graph.rope(q, THETA, HEAD_DIM);
        let k = graph.rope(k, THETA, HEAD_DIM);
        let attention = graph.causal_attention(q, k, v, HEADS, KV_HEADS, HEAD_DIM);
        let attention = projection(
            graph,
            model,
            &format!("{prefix}.attn_output.weight"),
            attention,
            HIDDEN,
            HIDDEN,
        )?;
        x = graph.add(x, attention);

        let post_norm = norm(graph, model, &format!("{prefix}.ffn_norm.weight"), HIDDEN)?;
        let h = graph.rms_norm(x, post_norm, EPS);
        let gate = projection(
            graph,
            model,
            &format!("{prefix}.ffn_gate.weight"),
            h,
            HIDDEN,
            FFN,
        )?;
        let up = projection(
            graph,
            model,
            &format!("{prefix}.ffn_up.weight"),
            h,
            HIDDEN,
            FFN,
        )?;
        let h = graph.swiglu(gate, up);
        let h = projection(
            graph,
            model,
            &format!("{prefix}.ffn_down.weight"),
            h,
            FFN,
            HIDDEN,
        )?;
        x = graph.add(x, h);
    }

    let flat = graph.reshape(x, &[sequence * HIDDEN]);
    let kept = graph.split_b(
        flat,
        1,
        drop as u32,
        (sequence - drop) as u32,
        HIDDEN as u32,
    );
    Ok(graph.reshape(kept, &[sequence - drop, HIDDEN]))
}

pub fn load_weights(session: &mut Session, model: &GgufModel) -> Result<(), Box<dyn Error>> {
    let names: Vec<String> = session
        .plan()
        .param_buffers
        .iter()
        .map(|(name, _)| name.clone())
        .collect();
    let mut loaded = 0;
    for name in names {
        let Some(tensor) = model.tensors.get(&name) else {
            // Optimizer-created concatenations are populated by uploading
            // their source tensors below.
            continue;
        };
        if tensor.dims.len() == 1 {
            session.set_parameter(&name, &tensor.to_f32()?);
        } else {
            match matrix_kind(tensor)? {
                MatrixKind::Dense => session.set_parameter(&name, &tensor.to_f32_rows()?),
                MatrixKind::Packed(_) => {
                    let (_, bytes) = tensor.to_packed()?;
                    session.set_parameter_packed(&name, &bytes);
                }
            }
        }
        loaded += 1;
    }
    let expected = LAYERS * 11;
    if loaded != expected {
        return Err(format!("loaded {loaded} text tensors, expected {expected}").into());
    }
    println!("loaded {loaded} quantized text encoder tensors");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    use meganeura::load::gguf::GgufValue;

    fn synthetic_model() -> GgufModel {
        let mut metadata = HashMap::new();
        metadata.insert(
            "general.architecture".to_owned(),
            GgufValue::String("qwen3vl".to_owned()),
        );
        let mut tensors = HashMap::new();
        let mut insert = |name: String, dims: Vec<usize>, ty: GgmlType| {
            tensors.insert(name, GgufTensor::new(dims, ty, Vec::new()));
        };
        for layer in 0..LAYERS {
            let prefix = format!("blk.{layer}");
            for suffix in ["attn_norm", "ffn_norm"] {
                insert(
                    format!("{prefix}.{suffix}.weight"),
                    vec![HIDDEN],
                    GgmlType::F32,
                );
            }
            for suffix in ["attn_q_norm", "attn_k_norm"] {
                insert(
                    format!("{prefix}.{suffix}.weight"),
                    vec![HEAD_DIM as usize],
                    GgmlType::F32,
                );
            }
            for (suffix, input, output) in [
                ("attn_q", HIDDEN, HIDDEN),
                ("attn_k", HIDDEN, KV_DIM),
                ("attn_v", HIDDEN, KV_DIM),
                ("attn_output", HIDDEN, HIDDEN),
                ("ffn_gate", HIDDEN, FFN),
                ("ffn_up", HIDDEN, FFN),
                ("ffn_down", FFN, HIDDEN),
            ] {
                insert(
                    format!("{prefix}.{suffix}.weight"),
                    vec![input, output],
                    GgmlType::Q4K,
                );
            }
        }
        GgufModel { metadata, tensors }
    }

    #[test]
    fn prompt_template_matches_pipeline_spelling() {
        let prompt = "a glass dragon";
        let system = format!("<|im_start|>system\n{SYSTEM}<|im_end|>\n");
        let full = format!("{system}<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n");
        assert!(full.starts_with("<|im_start|>system\nComprehend"));
        assert!(full.ends_with("<|im_start|>assistant\n"));
    }

    #[test]
    fn q4_text_graph_has_post_system_output_shape() {
        let model = synthetic_model();
        let mut graph = Graph::new();
        let output = build(&mut graph, &model, 7, 3).unwrap();
        assert_eq!(graph.node(output).ty.shape, [4, HIDDEN]);
    }

    #[test]
    fn token_lookup_only_decodes_requested_q8_rows() {
        let mut model = synthetic_model();
        let row_bytes = HIDDEN / 32 * 34;
        model.tensors.insert(
            "token_embd.weight".to_owned(),
            GgufTensor::new(vec![HIDDEN, 2], GgmlType::Q8_0, vec![0; row_bytes * 2]),
        );
        let got = token_embeddings(&model, &[1, 0]).unwrap();
        assert_eq!(got.len(), 2 * HIDDEN);
        assert!(got.iter().all(|value| *value == 0.0));
    }
}
