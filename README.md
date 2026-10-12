# mega-image

Meganeura-backed inference for the `qwen_image21` diffusion transformer in
`abenzerps/Qwen-Image-2.1-Uncensored-GGUF`.

The executable accepts a text prompt, downloads the required files through the
official `hf-hub` Rust package, encodes the prompt with a quantized Qwen3-VL
frontend, and runs the Qwen-Image 2.1 diffusion transformer. Downloads are
resumable and reused from the standard Hugging Face cache (`HF_HUB_CACHE`, or
`$HF_HOME/hub`). `HF_TOKEN` and `HF_ENDPOINT` are honored by the Hub client.

```sh
cargo run --release -- \
  --prompt "A glass dragon sleeping in an overgrown observatory" \
  --output latents.f32
```

The default 1024x1024 resolution constructs 4096 image tokens and requires a
substantial amount of GPU memory. For a first end-to-end smoke run, use 512x512
and one sampling step:

```sh
cargo run --release -- \
  --width 512 --height 512 --steps 1 \
  --prompt "A glass dragon sleeping in an overgrown observatory" \
  --output smoke.f32
```

Once the files are cached, add `--offline` to prohibit network access.

By default this resolves:

- `qwen-image-2.1-UC-Q4_K_M.gguf` from
  `abenzerps/Qwen-Image-2.1-Uncensored-GGUF`;
- the tokenizer from `Qwen/Qwen-Image-2.1`;
- the language-only `Qwen-Image-2.1-Text-Encoder-Q4_K_M.gguf` from
  `darrellbest/Qwen-Image-2.1-Text-Encoder-GGUF`.

The initial download is approximately 10.3 GB: 5.67 GB for the Q4_K_M text
encoder and 4.60 GB for the Q4_K_M diffusion transformer. The text encoder
runs first and is released before the diffusion transformer is constructed,
limiting peak GPU residency. Only the prompt's token-embedding rows are
dequantized; the full 622-million-value embedding table is never expanded on
the GPU.

Model setup is also host-memory intensive. With the currently pinned
Meganeura loader, allow roughly 20 GiB of available system memory while each
GGUF is copied into RAM and its GPU parameter buffers are populated. Reducing
the image dimensions lowers activation and attention memory, but not this
fixed weight-loading peak.

Useful overrides:

```sh
# Use an existing GGUF while still downloading/caching the text frontend.
cargo run --release -- --model /models/qwen-image.gguf --prompt "..."

# Fail rather than access the network; every required file must be cached.
cargo run --release -- --offline --prompt "..."

# Preserve the old precomputed-embedding path and skip the text encoder.
cargo run --release -- \
  --prompt-embeddings prompt.f32 \
  --output latents.f32
```

Run `mega-image --help` for repository, filename, resolution, step, and seed
options. A local `prompt.f32` is raw little-endian `f32`, shaped
`[tokens, 4096]`.

The current output is raw little-endian `f32`, shaped
`[height / 16 * width / 16, 64]`. The Qwen-Image 2.1 VAE decoder is not yet
implemented, so the executable does not currently write PNG files.

This crate pins a merged Meganeura `main` revision containing Qwen-Image's
three-axis RoPE and block-causal attention. The Qwen3-VL frontend reuses
Meganeura's lossless GGUF Q4_K, Q6_K, Q8_0, and dense-weight loaders.
