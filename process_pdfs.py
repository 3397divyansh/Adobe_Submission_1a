import os
import json
import fitz  # PyMuPDF
import numpy as np
from pathlib import Path
from onnx_helper import get_embedding

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Reference headings per level
reference_texts = {
    "H1": ["Introduction", "Overview", "Executive Summary"],
    "H2": ["Background", "Implementation", "Problem Statement"],
    "H3": ["Details", "Examples", "Results"]
}

# Precompute reference embeddings
reference_embeddings = {
    level: [get_embedding(text) for text in texts]
    for level, texts in reference_texts.items()
}

def classify_level(text_embedding):
    best_level = None
    best_score = -1

    for level, embeddings in reference_embeddings.items():
        for ref in embeddings:
            score = cosine_similarity(text_embedding, ref)
            if score > best_score:
                best_score = score
                best_level = level

    return best_level if best_score > 0.6 else None  # adjust threshold if needed

def extract_outline(pdf_path):
    doc = fitz.open(pdf_path)
    outline = []
    seen = set()

    for page_num, page in enumerate(doc, start=1):
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            if "lines" not in block:
                continue
            for line in block["lines"]:
                for span in line["spans"]:
                    text = span["text"].strip()
                    if not text or len(text) < 4 or text in seen:
                        continue
                    seen.add(text)

                    # Generate embedding for line
                    embedding = get_embedding(text)
                    level = classify_level(embedding)
                    if level:
                        outline.append({
                            "level": level,
                            "text": text,
                            "page": page_num
                        })

    title = doc.metadata.get("title") or pdf_path.stem
    return {
        "title": title.strip(),
        "outline": outline
    }

def process_pdfs():
    input_dir = Path("/app/input")
    output_dir = Path("/app/output")
    output_dir.mkdir(parents=True, exist_ok=True)

    pdf_files = list(input_dir.glob("*.pdf"))
    for pdf_file in pdf_files:
        result = extract_outline(pdf_file)
        output_file = output_dir / f"{pdf_file.stem}.json"
        with open(output_file, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Processed {pdf_file.name}")

if __name__ == "__main__":
    print("Running PDF heading extractor with ONNX model...")
    process_pdfs()
    print("Done.")
