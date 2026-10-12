mod hub;
mod model;
mod text_encoder;

use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};

use meganeura::Graph;
use meganeura::load::gguf::load_gguf;

struct Args {
    model: Option<PathBuf>,
    model_repo: String,
    model_file: String,
    base_repo: String,
    text_repo: String,
    text_file: String,
    prompt: Option<String>,
    prompt_embeddings: Option<PathBuf>,
    output: PathBuf,
    width: usize,
    height: usize,
    steps: usize,
    seed: u64,
    offline: bool,
}

fn usage() -> &'static str {
    "usage: mega-image --prompt TEXT [OPTIONS]\n\n\
     The default model, tokenizer, and Qwen3-VL text encoder are downloaded through\n\
     Hugging Face Hub and reused from its cache. The executable emits raw 64-channel\n\
     Qwen-Image 2.1 latents; VAE decoding is not yet included.\n\n\
     Inputs:\n\
       --prompt TEXT                  Encode TEXT with Qwen3-VL\n\
       --prompt-embeddings FILE       Use raw [tokens,4096] f32 instead\n\
       --model FILE                  Use a local transformer GGUF\n\
       --model-repo OWNER/NAME       GGUF repo (default: abenzerps/Qwen-Image-2.1-Uncensored-GGUF)\n\
       --model-file FILE             GGUF filename (default: qwen-image-2.1-UC-Q4_K_M.gguf)\n\
       --base-repo OWNER/NAME        Tokenizer repo (default: Qwen/Qwen-Image-2.1)\n\
       --text-repo OWNER/NAME        Text GGUF repository\n\
       --text-file FILE              Text GGUF filename (default: Q4_K_M)\n\
       --offline                     Resolve only from the Hugging Face cache\n\n\
     Sampling:\n\
       --output FILE                 Default: latents.f32\n\
       --width N --height N          Default: 1024x1024\n\
       --steps N                     Default: 40\n\
       --seed N                      Default: 42"
}

fn parse_args() -> Result<Args, Box<dyn Error>> {
    let values: Vec<String> = std::env::args().skip(1).collect();
    let optional = |flag: &str| -> Result<Option<String>, Box<dyn Error>> {
        let Some(index) = values.iter().position(|arg| arg == flag) else {
            return Ok(None);
        };
        Ok(Some(
            values
                .get(index + 1)
                .filter(|value| !value.starts_with("--"))
                .cloned()
                .ok_or_else(|| format!("missing value after {flag}"))?,
        ))
    };
    let prompt = optional("--prompt")?;
    let prompt_embeddings = optional("--prompt-embeddings")?.map(PathBuf::from);
    if prompt.is_some() == prompt_embeddings.is_some() {
        return Err("pass exactly one of --prompt or --prompt-embeddings".into());
    }
    Ok(Args {
        model: optional("--model")?.map(PathBuf::from),
        model_repo: optional("--model-repo")?.unwrap_or_else(|| hub::DEFAULT_MODEL_REPO.to_owned()),
        model_file: optional("--model-file")?.unwrap_or_else(|| hub::DEFAULT_MODEL_FILE.to_owned()),
        base_repo: optional("--base-repo")?.unwrap_or_else(|| hub::DEFAULT_BASE_REPO.to_owned()),
        text_repo: optional("--text-repo")?.unwrap_or_else(|| hub::DEFAULT_TEXT_REPO.to_owned()),
        text_file: optional("--text-file")?.unwrap_or_else(|| hub::DEFAULT_TEXT_FILE.to_owned()),
        prompt,
        prompt_embeddings,
        output: optional("--output")?
            .map(PathBuf::from)
            .unwrap_or_else(|| "latents.f32".into()),
        width: optional("--width")?.map_or(Ok(1024), |value| value.parse())?,
        height: optional("--height")?.map_or(Ok(1024), |value| value.parse())?,
        steps: optional("--steps")?.map_or(Ok(40), |value| value.parse())?,
        seed: optional("--seed")?.map_or(Ok(42), |value| value.parse())?,
        offline: values.iter().any(|value| value == "--offline"),
    })
}

fn read_f32(path: &Path) -> Result<Vec<f32>, Box<dyn Error>> {
    let bytes = fs::read(path)?;
    if !bytes.len().is_multiple_of(4) {
        return Err(format!("{} is not a raw f32 file", path.display()).into());
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect())
}

fn write_f32(path: &Path, values: &[f32]) -> Result<(), Box<dyn Error>> {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    values
        .iter()
        .for_each(|v| bytes.extend_from_slice(&v.to_le_bytes()));
    fs::write(path, bytes)?;
    Ok(())
}

fn positions(text: usize, h: usize, w: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity((text + h * w) * 3);
    for index in 0..text {
        out.extend([index as f32; 3]);
    }
    for y in 0..h {
        for x in 0..w {
            out.push(text as f32);
            out.push((y as i32 - (h as i32 - h as i32 / 2)) as f32);
            out.push((x as i32 - (w as i32 - w as i32 / 2)) as f32);
        }
    }
    out
}

fn time_embedding(t: f32) -> Vec<f32> {
    let mut out = Vec::with_capacity(256);
    for i in 0..128 {
        out.push((t * 1000.0 * 10_000.0f32.powf(-(i as f32) / 128.0)).cos());
    }
    for i in 0..128 {
        out.push((t * 1000.0 * 10_000.0f32.powf(-(i as f32) / 128.0)).sin());
    }
    out
}

fn normal_noise(len: usize, mut state: u64) -> Vec<f32> {
    let mut uniform = || {
        state = state.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64
    };
    let mut out = Vec::with_capacity(len);
    while out.len() < len {
        let u1 = uniform().max(f64::MIN_POSITIVE);
        let u2 = uniform();
        let r = (-2.0 * u1.ln()).sqrt();
        let a = std::f64::consts::TAU * u2;
        out.push((r * a.cos()) as f32);
        if out.len() < len {
            out.push((r * a.sin()) as f32);
        }
    }
    out
}

fn sigmas(steps: usize, image_tokens: usize) -> Vec<f32> {
    if steps == 1 {
        return vec![1.0, 0.0];
    }
    let base_len = 256.0;
    let max_len = 8192.0;
    let mu = 0.5 + (0.9 - 0.5) * (image_tokens as f32 - base_len) / (max_len - base_len);
    let e = mu.exp();
    let mut values: Vec<f32> = (0..steps)
        .map(|i| 1.0 - i as f32 * (1.0 - 1.0 / steps as f32) / (steps - 1).max(1) as f32)
        .map(|sigma| e / (e + (1.0 / sigma - 1.0)))
        .collect();
    // Qwen-Image 2.1's scheduler stretches the last sampled sigma to 0.02,
    // then appends the exact terminal zero used by the final Euler step.
    let scale = (1.0 - values[values.len() - 1]) / (1.0 - 0.02);
    values
        .iter_mut()
        .for_each(|sigma| *sigma = 1.0 - (1.0 - *sigma) / scale);
    values.push(0.0);
    values
}

fn encode_prompt(
    prompt: &str,
    tokenizer_repo: &str,
    text_repo: &str,
    text_file: &str,
    offline: bool,
) -> Result<Vec<f32>, Box<dyn Error>> {
    let files = hub::resolve_text_files(tokenizer_repo, text_repo, text_file, offline)?;
    let tokens = text_encoder::tokenize(&files.tokenizer, prompt)?;
    println!(
        "encoding {} prompt tokens (dropping {} system tokens)",
        tokens.ids.len(),
        tokens.drop
    );
    let output_tokens = tokens.ids.len() - tokens.drop;
    println!("loading {}", files.model.display());
    let text_model = load_gguf(&files.model)?;
    let token_embeddings = text_encoder::token_embeddings(&text_model, &tokens.ids)?;
    let mut graph = Graph::new();
    let output = text_encoder::build(&mut graph, &text_model, tokens.ids.len(), tokens.drop)?;
    graph.set_outputs(vec![output]);
    let mut session = meganeura::build(&graph, meganeura::SessionConfig::inference_from_env()).0;
    session.set_submission_chunks(1);
    text_encoder::load_weights(&mut session, &text_model)?;
    drop(text_model);
    session.set_input("token_embeddings", &token_embeddings);
    session.step();
    let mut embeddings = vec![0.0; output_tokens * text_encoder::HIDDEN];
    session.wait_read_output(0, &mut embeddings);
    Ok(embeddings)
}

fn run() -> Result<(), Box<dyn Error>> {
    let args = parse_args().map_err(|e| format!("{e}\n{}", usage()))?;
    if args.width % 32 != 0 || args.height % 32 != 0 || args.steps == 0 {
        return Err("width and height must be divisible by 32, and steps must be nonzero".into());
    }
    let prompt = match (&args.prompt, &args.prompt_embeddings) {
        (Some(prompt), None) => encode_prompt(
            prompt,
            &args.base_repo,
            &args.text_repo,
            &args.text_file,
            args.offline,
        )?,
        (None, Some(path)) => read_f32(path)?,
        _ => unreachable!("parse_args enforces one prompt input"),
    };
    if !prompt.len().is_multiple_of(model::HIDDEN) || prompt.is_empty() {
        return Err(
            "prompt embedding file must contain a nonzero multiple of 4096 f32 values".into(),
        );
    }
    let text_tokens = prompt.len() / model::HIDDEN;
    let (height_tokens, width_tokens) = (args.height / 16, args.width / 16);
    let model_path =
        hub::resolve_transformer(args.model, &args.model_repo, &args.model_file, args.offline)?;
    println!("loading {}", model_path.display());
    let gguf = load_gguf(&model_path)?;
    if gguf.architecture() != Some("qwen_image21") {
        return Err(format!(
            "expected qwen_image21 GGUF, found {:?}",
            gguf.architecture()
        )
        .into());
    }
    let mut graph = Graph::new();
    let built = model::build(&mut graph, &gguf, text_tokens, height_tokens, width_tokens)?;
    graph.set_outputs(vec![built.output]);
    println!(
        "compiling transformer for {text_tokens} text + {} image tokens",
        built.image_tokens
    );
    let mut session = meganeura::build(&graph, meganeura::SessionConfig::inference_from_env()).0;
    session.set_submission_chunks(1);
    println!("loading {} transformer tensors", gguf.tensors.len());
    model::load_weights(&mut session, &gguf)?;
    drop(gguf);
    session.set_input("prompt_embeddings", &prompt);
    session.set_input(
        "positions",
        &positions(text_tokens, height_tokens, width_tokens),
    );

    let mut latents = normal_noise(built.image_tokens * model::CHANNELS, args.seed);
    let schedule = sigmas(args.steps, built.image_tokens);
    let mut velocity = vec![0.0; latents.len()];
    for step in 0..args.steps {
        session.set_input("latents", &latents);
        session.set_input("time_embedding", &time_embedding(schedule[step]));
        session.step();
        session.wait_read_output(0, &mut velocity);
        let dt = schedule[step + 1] - schedule[step];
        latents
            .iter_mut()
            .zip(&velocity)
            .for_each(|(x, &v)| *x += dt * v);
        println!(
            "step {}/{} sigma={:.6}",
            step + 1,
            args.steps,
            schedule[step]
        );
    }
    write_f32(&args.output, &latents)?;
    println!(
        "wrote {} raw latent values to {}",
        latents.len(),
        args.output.display()
    );
    Ok(())
}

fn main() {
    if std::env::args().any(|arg| arg == "-h" || arg == "--help") {
        println!("{}", usage());
        return;
    }
    if let Err(error) = run() {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn axial_positions_match_qwen_centering() {
        let got = positions(2, 2, 2);
        assert_eq!(&got[..6], &[0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);
        assert_eq!(
            &got[6..],
            &[
                2.0, -1.0, -1.0, 2.0, -1.0, 0.0, 2.0, 0.0, -1.0, 2.0, 0.0, 0.0,
            ]
        );
    }

    #[test]
    fn scheduler_starts_at_one_and_uses_terminal_stretch() {
        let got = sigmas(4, 4096);
        assert_eq!(got.len(), 5);
        assert!((got[0] - 1.0).abs() < 1.0e-6);
        assert!((got[3] - 0.02).abs() < 1.0e-6);
        assert_eq!(got[4], 0.0);
        assert!(got.windows(2).all(|pair| pair[0] > pair[1]));
    }

    #[test]
    fn one_step_scheduler_goes_directly_to_zero() {
        assert_eq!(sigmas(1, 4096), [1.0, 0.0]);
    }

    #[test]
    fn zero_time_embedding_is_cos_then_sin() {
        let got = time_embedding(0.0);
        assert!(got[..128].iter().all(|&value| value == 1.0));
        assert!(got[128..].iter().all(|&value| value == 0.0));
    }
}
