from rag import load_store, retrieve


def main():
    index, chunks, model = load_store()
    while True:
        try:
            question = input("\nNhập câu hỏi (gõ exit để thoát): ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if question.lower() == "exit":
            break
        if not question:
            continue
        for rank, result in enumerate(retrieve(question, index, chunks, model), start=1):
            chunk = result["chunk"]
            print(f"\n[{rank}] {chunk['title']} | cosine: {result['score']:.4f}")
            if result["graph_topics"]:
                print("Chủ đề Neo4j:", ", ".join(result["graph_topics"]))
            print(chunk.get("url", ""))
            print(chunk["text"])


if __name__ == "__main__":
    main()
