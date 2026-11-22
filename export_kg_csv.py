import csv
import json

from chunk_documents import OUTPUT_FILE
from kg_topics import TOPICS_FILE, load_topics, match_topics


def write_csv(filename, fields, rows):
    target = TOPICS_FILE.with_name(filename)
    with target.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"{target.name}: {len(rows)} dòng")


def main():
    if not OUTPUT_FILE.is_file():
        raise SystemExit("Chạy python chunk_documents.py trước.")

    chunks = json.loads(OUTPUT_FILE.read_text(encoding="utf-8"))
    if not chunks:
        raise SystemExit("Không có đoạn tài liệu để xuất.")

    topics = load_topics()
    documents = {}
    links = []
    fields = ("chunk_id", "document_id", "heading", "section_index", "text")
    for chunk in chunks:
        document_id = chunk["document_id"]
        documents[document_id] = {
            field: chunk[field] for field in ("document_id", "title", "url")
        }
        tags = match_topics(f"{chunk['title']} {chunk['heading']}", topics)
        links.extend({"chunk_id": chunk["chunk_id"], "topic_id": tag} for tag in tags)

    write_csv("documents.csv", ("document_id", "title", "url"), list(documents.values()))
    write_csv("chunks.csv", fields, [{field: c[field] for field in fields} for c in chunks])
    write_csv("chunk_topics.csv", ("chunk_id", "topic_id"), links)


if __name__ == "__main__":
    main()
