use candle_core::{Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config};
use std::error::Error;
use std::fs::File;
use std::path::Path;
use tokenizers::Tokenizer;
use tracing::info;

fn download_file(url: &str, dest: &str) -> Result<String, Box<dyn Error>> {
    if !Path::new(dest).exists() {
        info!("Downloading {}...", dest);
        let resp = ureq::get(url).call()?;
        let mut out = File::create(dest)?;
        std::io::copy(&mut resp.into_reader(), &mut out)?;
    }
    Ok(dest.to_string())
}

pub fn load_model() -> Result<(BertModel, Tokenizer), Box<dyn Error>> {
    info!("Fetching safetensors and config...");

    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let data_dir = format!("{}/data", manifest_dir);

    std::fs::create_dir_all(&data_dir)?;

    let config_path = download_file(
        "https://huggingface.co/BAAI/bge-base-en-v1.5/resolve/main/config.json",
        &format!("{}/config.json", data_dir),
    )?;
    let tokenizer_path = download_file(
        "https://huggingface.co/BAAI/bge-base-en-v1.5/resolve/main/tokenizer.json",
        &format!("{}/tokenizer.json", data_dir),
    )?;
    let weights_path = download_file(
        "https://huggingface.co/BAAI/bge-base-en-v1.5/resolve/main/model.safetensors",
        &format!("{}/model.safetensors", data_dir),
    )?;

    info!("Building Tokenizer and Tensor Neural Network...");

    let config = std::fs::read_to_string(config_path)?;
    let config: Config = serde_json::from_str(&config)?;

    let mut tokenizer =
        Tokenizer::from_file(tokenizer_path).map_err(|e| Box::<dyn Error>::from(e.to_string()))?;

    tokenizer.with_padding(Some(tokenizers::PaddingParams {
        strategy: tokenizers::PaddingStrategy::BatchLongest,
        ..Default::default()
    }));

    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[weights_path], candle_core::DType::F32, &Device::Cpu)?
    };

    let model = BertModel::load(vb, &config)?;

    Ok((model, tokenizer))
}

fn normalize_vector(mut vec: Vec<f32>) -> Vec<f32> {
    let magnitude = vec.iter().map(|&x| x * x).sum::<f32>().sqrt();
    if magnitude > 0.0 {
        vec.iter_mut().for_each(|x| *x /= magnitude);
    }
    vec
}

pub fn get_embeddings(
    sentences: &[String],
    tokenizer: &Tokenizer,
    model: &BertModel,
) -> Result<Vec<Vec<f32>>, Box<dyn Error>> {
    let inputs: Vec<&str> = sentences.iter().map(|s| s.as_str()).collect();
    let tokens = tokenizer
        .encode_batch(inputs, true)
        .map_err(|e| Box::<dyn Error>::from(e.to_string()))?;

    let token_ids = tokens
        .iter()
        .map(|t| Tensor::new(t.get_ids(), &Device::Cpu))
        .collect::<Result<Vec<_>, _>>()?;
    let token_ids = Tensor::stack(&token_ids, 0)?;

    let token_type_ids = token_ids.zeros_like()?;
    let embeddings = model.forward(&token_ids, &token_type_ids, None)?;

    let cls_embeddings = embeddings.i((.., 0, ..))?;

    let raw_vecs = cls_embeddings.to_vec2::<f32>()?;
    let normalized_vecs = raw_vecs.into_iter().map(normalize_vector).collect();

    Ok(normalized_vecs)
}
