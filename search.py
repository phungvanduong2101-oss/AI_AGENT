import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent
STORE_DIR = BASE_DIR / "data/vector_store"


def main():
    with (STORE_DIR / "config.json").open("r", encoding="utf-8") as f:
        config = json.load(f)

    with (STORE_DIR / "metadata.json").open("r", encoding="utf-8") as f:
        chunks = json.load(f)

    index = faiss.read_index(str(STORE_DIR / "index.faiss"))

    if index.ntotal != len(chunks):
        raise ValueError("Số vector và số chunks không khớp")

    model = SentenceTransformer(config["model_name"])

    while True:
        question = input("\nNhập câu hỏi (gõ exit để thoát): ").strip()

        if question.lower() == "exit":
            break

        if not question:
            continue

        vector = model.encode(
            [question],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        vector = np.ascontiguousarray(vector, dtype=np.float32)

        top_k = min(3, index.ntotal)
        scores, positions = index.search(vector, top_k)

        for rank, (score, position) in enumerate(
            zip(scores[0], positions[0]), start=1
        ):
            if position < 0:
                continue

            chunk = chunks[int(position)]

            print(f"\n[{rank}] {chunk['title']}")
            print(f"Mục: {chunk.get('heading', '')}")
            print(f"Chunk: {chunk['chunk_id']}")
            print(f"Similarity: {float(score):.4f}")
            print(f"URL: {chunk.get('url', '')}")
            print(f"Nội dung:\n{chunk['text']}")


if __name__ == "__main__":
    main()