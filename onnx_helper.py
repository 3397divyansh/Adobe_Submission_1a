import onnxruntime as ort
import numpy as np
from transformers import AutoTokenizer

MODEL_PATH = "./model"
TOKENIZER_PATH = "./model/local_model"  # path to saved tokenizer

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)

def get_embedding(text):
    inputs = tokenizer(text, return_tensors="np", padding="max_length", truncation=True, max_length=128)
    ort_session = ort.InferenceSession(f"{MODEL_PATH}/model.onnx")
    outputs = ort_session.run(None, dict(inputs))
    return np.mean(outputs[0], axis=1)[0]
