import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "data/chunks.json"
OUTPUT_DIR = BASE_DIR / "data/vector_store"

MODEL_NAME = "intfloat/multilingual-e5-small"

def main():
    with INPUT_FILE.open("r", encoding="utf-8") as f:
        chunks = json.load(f)

    if not chunks:
        raise ValueError("Danh sách chunks rỗng")

    texts = [
        "passage: "
        + chunk.get("title", "")
        + "\n"
        + chunk.get("heading", "")
        + "\n"
        + chunk["text"]
        for chunk in chunks
    ]

    print("Đang tải model...")
    model = SentenceTransformer(MODEL_NAME)

    print(f"Đang tạo embedding cho {len(texts)} chunks...")
    vectors = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    vectors = np.ascontiguousarray(vectors, dtype=np.float32)

    if vectors.shape != (len(chunks), 384):
        raise ValueError(f"Kích thước vector không đúng: {vectors.shape}")

    index = faiss.IndexFlatIP(384)
    index.add(vectors)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(OUTPUT_DIR / "index.faiss"))

    with (OUTPUT_DIR / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    with (OUTPUT_DIR / "config.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "model_name": MODEL_NAME,
                "dimension": 384,
                "normalized": True,
                "text_field": "text",
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"\nĐã lưu {index.ntotal} vectors, mỗi vector 384 chiều.")
    print(f"Thư mục: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()