use std::error::Error;
use std::path::PathBuf;

use hf_hub::HFClientSync;

pub const DEFAULT_MODEL_REPO: &str = "abenzerps/Qwen-Image-2.1-Uncensored-GGUF";
pub const DEFAULT_MODEL_FILE: &str = "qwen-image-2.1-UC-Q4_K_M.gguf";
pub const DEFAULT_BASE_REPO: &str = "Qwen/Qwen-Image-2.1";
pub const DEFAULT_TEXT_REPO: &str = "darrellbest/Qwen-Image-2.1-Text-Encoder-GGUF";
pub const DEFAULT_TEXT_FILE: &str = "Qwen-Image-2.1-Text-Encoder-Q4_K_M.gguf";

pub struct TextFiles {
    pub tokenizer: PathBuf,
    pub model: PathBuf,
}

fn split_repo(repo: &str) -> Result<(&str, &str), Box<dyn Error>> {
    repo.split_once('/')
        .ok_or_else(|| format!("Hugging Face repository must be OWNER/NAME, got `{repo}`").into())
}

fn download(
    client: &HFClientSync,
    repo_id: &str,
    filename: &str,
    offline: bool,
) -> Result<PathBuf, Box<dyn Error>> {
    let (owner, name) = split_repo(repo_id)?;
    println!("resolving hf://{repo_id}/{filename}");
    Ok(client
        .model(owner, name)
        .download_file()
        .filename(filename)
        .local_files_only(offline)
        .send()?)
}

pub fn resolve_transformer(
    local: Option<PathBuf>,
    repo: &str,
    filename: &str,
    offline: bool,
) -> Result<PathBuf, Box<dyn Error>> {
    if let Some(path) = local {
        return Ok(path);
    }
    let client = HFClientSync::new()?;
    download(&client, repo, filename, offline)
}

pub fn resolve_text_files(
    tokenizer_repo: &str,
    text_repo: &str,
    text_file: &str,
    offline: bool,
) -> Result<TextFiles, Box<dyn Error>> {
    let client = HFClientSync::new()?;
    let tokenizer = download(&client, tokenizer_repo, "processor/tokenizer.json", offline)?;
    let model = download(&client, text_repo, text_file, offline)?;
    Ok(TextFiles { tokenizer, model })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repo_ids_require_an_owner() {
        assert!(split_repo("Qwen/Qwen-Image-2.1").is_ok());
        assert!(split_repo("Qwen-Image-2.1").is_err());
    }
}
