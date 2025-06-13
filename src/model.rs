use serde::Deserialize;

#[derive(Debug, Deserialize, Clone)]
pub struct LayerInfo {
    #[serde(rename = "type")]
    pub layer_type: String,
    pub input_dim: Option<usize>,
    pub output_dim: Option<usize>,
    pub activation: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct ModelInfo {
    pub layers: Vec<LayerInfo>,
}

#[derive(Debug, Deserialize)]
pub struct ForwardPassResult {
    pub activations: Vec<Vec<f32>>,
}
