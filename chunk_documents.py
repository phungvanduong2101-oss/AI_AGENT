import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "data/documents"
OUTPUT_FILE = BASE_DIR / "data/chunks.json"

CHUNK_SIZE = 150
CHUNK_OVERLAP = 30


def chunk_text(text):
    if CHUNK_SIZE <= 0:
        raise ValueError("CHUNK_SIZE phải lớn hơn 0")

    if not 0 <= CHUNK_OVERLAP < CHUNK_SIZE:
        raise ValueError(
            "CHUNK_OVERLAP phải từ 0 đến nhỏ hơn CHUNK_SIZE"
        )

    words = text.split()
    chunks = []
    step = CHUNK_SIZE - CHUNK_OVERLAP

    for start in range(0, len(words), step):
        end = start + CHUNK_SIZE
        chunks.append(" ".join(words[start:end]))
        if end >= len(words):
            break
    return chunks


def main():
    files = sorted(INPUT_DIR.glob("*.json"))

    if not files:
        raise ValueError(
            f"Không tìm thấy file JSON trong {INPUT_DIR}"
        )

    all_chunks = []
    seen_ids = set()

    for file in files:
        with file.open("r", encoding="utf-8-sig") as f:
            document = json.load(f)

        document_id = str(
            document.get("document_id", "")
        ).strip()
        title = document.get("title", "").strip()
        url = document.get("url", "").strip()

        if not document_id:
            raise ValueError(
                f"{file.name}: thiếu document_id"
            )

        if document_id in seen_ids:
            raise ValueError(
                f"{file.name}: trùng document_id {document_id}"
            )

        seen_ids.add(document_id)

        if not url:
            print(
                f"Cảnh báo: {file.name} chưa có URL nguồn"
            )

        chunk_number = 0

        for section_index, section in enumerate(
            document.get("sections", []),
            start=1,
        ):
            heading = section.get("heading", "").strip()
            text = section.get("text", "").strip()

            if not text:
                continue

            for content in chunk_text(text):
                chunk_number += 1

                all_chunks.append({
                    "chunk_id": (
                        f"{document_id}_c{chunk_number:04d}"
                    ),
                    "document_id": document_id,
                    "title": title,
                    "heading": heading,
                    "section_index": section_index,
                    "url": url,
                    "text": content,
                })

        print(f"{file.name}: {chunk_number} chunks")

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        json.dump(
            all_chunks,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"\nĐã xử lý {len(files)} tài liệu")
    print(f"Chunk size: {CHUNK_SIZE} từ")
    print(f"Overlap: {CHUNK_OVERLAP} từ")
    print(f"Tổng số chunks: {len(all_chunks)}")
    print(f"Đã lưu: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()