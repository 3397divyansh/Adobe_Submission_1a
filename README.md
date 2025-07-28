# Challenge 1a — PDF Outline Extraction

## Objective

Extract a structured outline from PDFs by identifying **headings and subheadings**, and output them in JSON format.

---

## Folder Structure

```
Challenge_1a/
├── model/                    # ONNX model directory
│   └── model.onnx
├── sample_dataset/
│   ├── pdfs/                 # Input PDFs for testing
│   └── outputs/              # JSON output files
├── process_pdfs.py          # Main processing script
├── onnx_helper.py           # Loads ONNX model and gets embeddings
├── requirements.txt         # Dependencies
├── Dockerfile               # Docker setup
└── README.md
```

---

## Approach

1. **Text Extraction**:

   - Extract text blocks from PDF using `PyMuPDF`.
   - Maintain line-level layout, size, and font information.

2. **Embedding + Classification**:

   - Use the `all-MiniLM-L6-v2` sentence transformer converted to ONNX.
   - Generate embeddings for each line.
   - Match each line’s embedding against a fixed set of reference embeddings like:
     - "Heading 1", "Heading 2", "Subheading", "Paragraph"
   - Use cosine similarity to classify.

3. **Structure Building**:

   - Based on similarity and confidence threshold, assign heading levels.
   - Group headings hierarchically (H1 > H2 > H3).

4. **Output**:

   - Output structured outline as JSON with heading levels and page numbers.

---

## Running with Docker

```bash
docker build -t pdf-processor .
docker run --rm \
  -v "$(pwd)/sample_dataset/pdfs:/app/input:ro" \
  -v "$(pwd)/sample_dataset/outputs:/app/output" \
  pdf-processor
```

---

## Output Format

```json
{
  "filename": "example.pdf",
  "outline": [
    {
      "title": "Introduction",
      "level": 1,
      "page": 1
    },
    {
      "title": "Subtopic",
      "level": 2,
      "page": 2
    }
  ]
}
```

---