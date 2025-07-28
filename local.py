# from transformers import AutoTokenizer, AutoModel
# model_name = "sentence-transformers/all-MiniLM-L6-v2"

# # Save locally
# AutoTokenizer.from_pretrained(model_name).save_pretrained("./local_model")
# AutoModel.from_pretrained(model_name).save_pretrained("./local_model")
import onnxruntime as ort
session = ort.InferenceSession("model/model.onnx")
print("✅ Model loaded successfully.")
