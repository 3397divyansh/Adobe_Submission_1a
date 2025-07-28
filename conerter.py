from transformers import AutoTokenizer, AutoModel
from transformers.onnx import export
from transformers.onnx.features import FeaturesManager
from pathlib import Path

model_name = "sentence-transformers/all-MiniLM-L6-v2"
feature = "default"

onnx_path = Path("model")
onnx_path.mkdir(exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

model_kind, model_onnx_config = FeaturesManager.check_supported_model_or_raise(model, feature=feature)
onnx_config = model_onnx_config(model.config)

export(
    preprocessor=tokenizer,
    model=model,
    config=onnx_config,
    opset=12,
    output=onnx_path / "model.onnx"
)

print("✅ Exported ONNX model to", onnx_path / "model.onnx")
