use std::error::Error;

use meganeura::graph::{DType, NodeId};
use meganeura::load::gguf::{GgmlType, GgufModel, GgufTensor};
use meganeura::{Graph, Session};

pub const HIDDEN: usize = 4096;
pub const HEADS: u32 = 32;
pub const HEAD_DIM: u32 = 128;
pub const CHANNELS: usize = 64;
pub const LAYERS: usize = 32;
const FFN: usize = HIDDEN * 3;
const EPS: f32 = 1.0e-6;

#[derive(Clone, Copy)]
enum MatrixKind {
    Dense,
    Packed(DType),
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
    parameter_name: &str,
    source_name: &str,
    k: usize,
    n: usize,
) -> Result<(NodeId, MatrixKind), Box<dyn Error>> {
    let tensor = model
        .tensors
        .get(source_name)
        .ok_or_else(|| format!("GGUF has no tensor `{source_name}`"))?;
    let source_n = if source_name == "modulation.1.weight" {
        HIDDEN * 4
    } else {
        n
    };
    if tensor.dims.as_slice() != [k, source_n] {
        return Err(format!(
            "`{source_name}` has dimensions {:?}, expected [{k}, {source_n}]",
            tensor.dims
        )
        .into());
    }
    let kind = matrix_kind(tensor)?;
    let node = match kind {
        MatrixKind::Dense => graph.parameter(parameter_name, &[n, k]),
        MatrixKind::Packed(DType::Q40) => graph.parameter_q40(parameter_name, &[k, n]),
        MatrixKind::Packed(DType::Q4_0) => graph.parameter_q4(parameter_name, &[k, n]),
        MatrixKind::Packed(DType::Q8_0) => graph.parameter_q8(parameter_name, &[k, n]),
        MatrixKind::Packed(DType::Q4K) => graph.parameter_q4k(parameter_name, &[k, n]),
        MatrixKind::Packed(DType::Q5K) => graph.parameter_q5k(parameter_name, &[k, n]),
        MatrixKind::Packed(DType::Q6K) => graph.parameter_q6k(parameter_name, &[k, n]),
        MatrixKind::Packed(DType::Q3K) => graph.parameter_q3k(parameter_name, &[k, n]),
        MatrixKind::Packed(other) => {
            return Err(format!("unsupported matrix dtype {other:?}").into());
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
    k: usize,
    n: usize,
) -> Result<NodeId, Box<dyn Error>> {
    let matrix = declare_matrix(graph, model, name, name, k, n)?;
    Ok(project(graph, x, matrix))
}

fn vstack(graph: &mut Graph, a: NodeId, b: NodeId, a_rows: usize, b_rows: usize) -> NodeId {
    let a = graph.reshape(a, &[a_rows * HIDDEN]);
    let b = graph.reshape(b, &[b_rows * HIDDEN]);
    let joined = graph.concat(a, b, 1, a_rows as u32, b_rows as u32, HIDDEN as u32);
    graph.reshape(joined, &[a_rows + b_rows, HIDDEN])
}

fn split_rows(graph: &mut Graph, x: NodeId, a_rows: usize, b_rows: usize) -> (NodeId, NodeId) {
    let flat_a = graph.split_a(x, 1, a_rows as u32, b_rows as u32, HIDDEN as u32);
    let flat_b = graph.split_b(x, 1, a_rows as u32, b_rows as u32, HIDDEN as u32);
    (
        graph.reshape(flat_a, &[a_rows, HIDDEN]),
        graph.reshape(flat_b, &[b_rows, HIDDEN]),
    )
}

fn affine_free_norm(graph: &mut Graph, x: NodeId, one: NodeId, zero: NodeId) -> NodeId {
    graph.layer_norm(x, one, zero, EPS)
}

fn modulate(graph: &mut Graph, x: NodeId, scale: NodeId, one_row: NodeId) -> NodeId {
    let scale = graph.add(scale, one_row);
    let scale = graph.reshape(scale, &[HIDDEN]);
    graph.bias_mul(x, scale)
}

fn gated(graph: &mut Graph, x: NodeId, gate: NodeId) -> NodeId {
    let gate = graph.tanh(gate);
    let gate = graph.reshape(gate, &[HIDDEN]);
    graph.bias_mul(x, gate)
}

pub struct Built {
    pub output: NodeId,
    pub image_tokens: usize,
}

pub fn build(
    graph: &mut Graph,
    model: &GgufModel,
    text_tokens: usize,
    height_tokens: usize,
    width_tokens: usize,
) -> Result<Built, Box<dyn Error>> {
    let image_tokens = height_tokens * width_tokens;
    let text_input = graph.input("prompt_embeddings", &[text_tokens, HIDDEN]);
    let latent_input = graph.input("latents", &[image_tokens, CHANNELS]);
    let positions = graph.input_u32("positions", &[text_tokens + image_tokens, 3]);
    let time_input = graph.input("time_embedding", &[1, 256]);

    let text_norm_source = model
        .tensors
        .get("txt_in.text_norm.weight")
        .ok_or("GGUF has no `txt_in.text_norm.weight`")?;
    if text_norm_source.dims.as_slice() != [HIDDEN] {
        return Err("text norm has the wrong shape".into());
    }
    let text_norm = graph.parameter("txt_in.text_norm.weight", &[HIDDEN]);
    let mut text = graph.rms_norm(text_input, text_norm, EPS);
    text = projection(graph, model, "txt_in.in_layer.weight", text, HIDDEN, HIDDEN)?;
    text = graph.gelu(text);
    text = projection(
        graph,
        model,
        "txt_in.out_layer.weight",
        text,
        HIDDEN,
        HIDDEN,
    )?;
    let mut image = projection(
        graph,
        model,
        "img_in.weight",
        latent_input,
        CHANNELS,
        HIDDEN,
    )?;

    let time1 = declare_matrix(
        graph,
        model,
        "time_text_embed.timestep_embedder.linear_1.weight",
        "time_text_embed.timestep_embedder.linear_1.weight",
        256,
        HIDDEN,
    )?;
    let time2 = declare_matrix(
        graph,
        model,
        "time_text_embed.timestep_embedder.linear_2.weight",
        "time_text_embed.timestep_embedder.linear_2.weight",
        HIDDEN,
        HIDDEN,
    )?;
    let zero_time = {
        let mut values = vec![0.0; 256];
        values[..128].fill(1.0);
        graph.constant(values, &[1, 256])
    };
    let embed_time = |g: &mut Graph, x, w1, w2| {
        let x = project(g, x, w1);
        let x = g.silu(x);
        project(g, x, w2)
    };
    let time = embed_time(graph, time_input, time1, time2);
    let time_zero = embed_time(graph, zero_time, time1, time2);

    let mut mods_time = Vec::with_capacity(4);
    let mut mods_zero = Vec::with_capacity(4);
    for index in 0..4 {
        let name = format!("modulation.1.weight#{index}");
        let matrix = declare_matrix(graph, model, &name, "modulation.1.weight", HIDDEN, HIDDEN)?;
        mods_time.push(project(graph, time, matrix));
        mods_zero.push(project(graph, time_zero, matrix));
    }

    let one = graph.constant(vec![1.0; HIDDEN], &[HIDDEN]);
    let zero = graph.constant(vec![0.0; HIDDEN], &[HIDDEN]);
    let one_row = graph.constant(vec![1.0; HIDDEN], &[1, HIDDEN]);

    for layer in 0..LAYERS {
        let prefix = format!("transformer_blocks.{layer}");
        let text_h = affine_free_norm(graph, text, one, zero);
        let image_h = affine_free_norm(graph, image, one, zero);
        let text_h = modulate(graph, text_h, mods_zero[0], one_row);
        let image_h = modulate(graph, image_h, mods_time[0], one_row);
        let joint = vstack(graph, text_h, image_h, text_tokens, image_tokens);

        let q = projection(
            graph,
            model,
            &format!("{prefix}.attn.to_q.weight"),
            joint,
            HIDDEN,
            HIDDEN,
        )?;
        let k = projection(
            graph,
            model,
            &format!("{prefix}.attn.to_k.weight"),
            joint,
            HIDDEN,
            HIDDEN,
        )?;
        let v = projection(
            graph,
            model,
            &format!("{prefix}.attn.to_v.weight"),
            joint,
            HIDDEN,
            HIDDEN,
        )?;
        let q_norm = graph.parameter(
            &format!("{prefix}.attn.norm_q.weight"),
            &[HEAD_DIM as usize],
        );
        let k_norm = graph.parameter(
            &format!("{prefix}.attn.norm_k.weight"),
            &[HEAD_DIM as usize],
        );
        let rows = text_tokens + image_tokens;
        let q = graph.reshape(q, &[rows * HEADS as usize, HEAD_DIM as usize]);
        let q = graph.rms_norm(q, q_norm, EPS);
        let q = graph.reshape(q, &[rows, HIDDEN]);
        let k = graph.reshape(k, &[rows * HEADS as usize, HEAD_DIM as usize]);
        let k = graph.rms_norm(k, k_norm, EPS);
        let k = graph.reshape(k, &[rows, HIDDEN]);
        let q = graph.axial_rope_with_positions(q, 10_000.0, positions, HEAD_DIM, [16, 56, 56]);
        let k = graph.axial_rope_with_positions(k, 10_000.0, positions, HEAD_DIM, [16, 56, 56]);
        let attn =
            graph.block_causal_attention(q, k, v, HEADS, HEADS, HEAD_DIM, text_tokens as u32);
        let attn = projection(
            graph,
            model,
            &format!("{prefix}.attn.to_out.0.weight"),
            attn,
            HIDDEN,
            HIDDEN,
        )?;
        let (text_attn, image_attn) = split_rows(graph, attn, text_tokens, image_tokens);
        let text_attn = gated(graph, text_attn, mods_zero[1]);
        let image_attn = gated(graph, image_attn, mods_time[1]);
        text = graph.add(text, text_attn);
        image = graph.add(image, image_attn);

        let text_h = affine_free_norm(graph, text, one, zero);
        let image_h = affine_free_norm(graph, image, one, zero);
        let text_h = modulate(graph, text_h, mods_zero[2], one_row);
        let image_h = modulate(graph, image_h, mods_time[2], one_row);
        let gate_weight = declare_matrix(
            graph,
            model,
            &format!("{prefix}.img_mlp.gate_layer.weight"),
            &format!("{prefix}.img_mlp.gate_layer.weight"),
            HIDDEN,
            FFN,
        )?;
        let up_weight = declare_matrix(
            graph,
            model,
            &format!("{prefix}.img_mlp.proj.weight"),
            &format!("{prefix}.img_mlp.proj.weight"),
            HIDDEN,
            FFN,
        )?;
        let out_weight = declare_matrix(
            graph,
            model,
            &format!("{prefix}.img_mlp.out.weight"),
            &format!("{prefix}.img_mlp.out.weight"),
            FFN,
            HIDDEN,
        )?;
        let text_gate = project(graph, text_h, gate_weight);
        let text_up = project(graph, text_h, up_weight);
        let text_mlp = graph.swiglu(text_gate, text_up);
        let text_mlp = project(graph, text_mlp, out_weight);
        let image_gate = project(graph, image_h, gate_weight);
        let image_up = project(graph, image_h, up_weight);
        let image_mlp = graph.swiglu(image_gate, image_up);
        let image_mlp = project(graph, image_mlp, out_weight);
        let text_mlp = gated(graph, text_mlp, mods_zero[3]);
        let image_mlp = gated(graph, image_mlp, mods_time[3]);
        text = graph.add(text, text_mlp);
        image = graph.add(image, image_mlp);
    }

    image = affine_free_norm(graph, image, one, zero);
    let final_scale = declare_matrix(
        graph,
        model,
        "norm_out.linear.weight",
        "norm_out.linear.weight",
        HIDDEN,
        HIDDEN,
    )?;
    let final_scale = project(graph, time, final_scale);
    image = modulate(graph, image, final_scale, one_row);
    let output = projection(graph, model, "proj_out.weight", image, HIDDEN, CHANNELS)?;
    Ok(Built {
        output,
        image_tokens,
    })
}

type SourceRows<'a> = (&'a GgufTensor, Option<std::ops::Range<usize>>);

fn source_and_rows<'a>(model: &'a GgufModel, name: &str) -> Result<SourceRows<'a>, Box<dyn Error>> {
    if let Some((source, suffix)) = name.split_once('#') {
        let index: usize = suffix.parse()?;
        let tensor = model
            .tensors
            .get(source)
            .ok_or("missing modulation tensor")?;
        return Ok((tensor, Some(index * HIDDEN..(index + 1) * HIDDEN)));
    }
    Ok((
        model
            .tensors
            .get(name)
            .ok_or_else(|| format!("missing tensor `{name}`"))?,
        None,
    ))
}

pub fn load_weights(session: &mut Session, model: &GgufModel) -> Result<(), Box<dyn Error>> {
    let names: Vec<String> = session
        .plan()
        .param_buffers
        .iter()
        .map(|(name, _)| name.clone())
        .collect();
    for name in names {
        // Optimizer-created concatenations are filled incrementally when
        // their original source parameters are uploaded below.
        let Ok((tensor, rows)) = source_and_rows(model, &name) else {
            continue;
        };
        if tensor.dims.len() == 1 {
            let mut values = tensor.to_f32()?;
            if name == "txt_in.text_norm.weight" {
                values.iter_mut().for_each(|value| *value += 1.0);
            }
            session.set_parameter(&name, &values);
            continue;
        }
        match matrix_kind(tensor)? {
            MatrixKind::Dense => {
                let values = tensor.to_f32_rows()?;
                let values = match rows {
                    Some(rows) => values[rows.start * HIDDEN..rows.end * HIDDEN].to_vec(),
                    None => values,
                };
                session.set_parameter(&name, &values);
            }
            MatrixKind::Packed(_) => {
                let (_, bytes) = tensor.to_packed()?;
                let selected: &[u8] = match rows {
                    Some(rows) => {
                        let row_bytes = bytes.len() / tensor.dims[1];
                        &bytes[rows.start * row_bytes..rows.end * row_bytes]
                    }
                    None => &bytes,
                };
                session.set_parameter_packed(&name, selected);
            }
        }
    }
    Ok(())
}
