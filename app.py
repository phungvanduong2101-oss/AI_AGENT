import os
import threading
from pathlib import Path
from time import perf_counter
from urllib.parse import urlparse

import streamlit as st
from dotenv import load_dotenv
from google import genai

import rag


BASE_DIR = Path(__file__).resolve().parent
STORE_DIR = BASE_DIR / "data/vector_store"

load_dotenv(BASE_DIR / ".env", override=True)

st.set_page_config(
    page_title="Diabetes RAG",
    page_icon="📚",
    layout="wide",
)


def store_version():
    """Đổi cache khi các tệp chỉ mục thay đổi."""
    versions = []

    for name in ("config.json", "metadata.json", "index.faiss"):
        file = STORE_DIR / name

        if not file.is_file():
            raise FileNotFoundError(
                f"Thiếu {file}. Hãy chạy build_index.py trước."
            )

        info = file.stat()
        versions.append((name, info.st_mtime_ns, info.st_size))

    return tuple(versions)


@st.cache_resource(show_spinner=False, max_entries=2)
def load_resources(version):
    index, chunks, model = rag.load_store()

    # Model được dùng chung giữa các lượt chạy giao diện.
    lock = threading.Lock()

    return index, chunks, model, lock


def show_source_url(url):
    if not isinstance(url, str):
        return

    parsed = urlparse(url)

    if parsed.scheme in ("http", "https") and parsed.netloc:
        st.link_button("Mở tài liệu gốc", url)


def show_sources(results):
    for number, result in enumerate(results, start=1):
        chunk = result["chunk"]
        score = result["score"]

        title = chunk.get("title") or "Tài liệu không có tiêu đề"

        with st.expander(
            f"[{number}] {title} — Similarity: {score:.4f}"
        ):
            st.write("**Mục:**", chunk.get("heading", ""))
            st.write("**Chunk ID:**", chunk.get("chunk_id", ""))
            st.write(
                "**Document ID:**",
                chunk.get("document_id", ""),
            )

            if result.get("graph_topics"):
                st.write("**Chủ đề Neo4j:**", ", ".join(result["graph_topics"]))

            show_source_url(chunk.get("url", ""))

            st.markdown("**Nội dung đoạn:**")
            st.write(chunk.get("text", ""))


def show_message(message):
    with st.chat_message(message["role"]):
        if message.get("error"):
            st.error(message["content"])
        else:
            st.markdown(message["content"])

        if "elapsed" in message:
            st.caption(
                f"Thời gian xử lý: {message['elapsed']:.2f} giây"
            )

        results = message.get("sources", [])

        if results:
            with st.expander(
                f"Các đoạn đã truy xuất ({len(results)})"
            ):
                show_sources(results)


def chat_page(index, chunks, model, lock, top_k):
    st.title("Hỏi đáp Diabetes RAG")
    st.caption(
        "Tra cứu kiến thức trong kho tài liệu. "
        "Mỗi câu hỏi được truy xuất độc lập."
    )

    for message in st.session_state.messages:
        show_message(message)

    question = st.chat_input(
        "Nhập câu hỏi về bệnh tiểu đường..."
    )

    if not question or not question.strip():
        return

    question = question.strip()

    user_message = {
        "role": "user",
        "content": question,
    }

    st.session_state.messages.append(user_message)
    show_message(user_message)

    results = []
    started = perf_counter()

    try:
        with st.spinner("Đang truy xuất và tạo câu trả lời..."):
            with lock:
                results = rag.retrieve(
                    question,
                    index,
                    chunks,
                    model,
                )[:top_k]

            api_key = os.getenv("GEMINI_API_KEY", "").strip()
            llm_model = os.getenv("GEMINI_MODEL", "").strip()

            if not api_key:
                raise ValueError(
                    "Thiếu GEMINI_API_KEY trong file .env."
                )

            if not llm_model:
                raise ValueError(
                    "Thiếu GEMINI_MODEL trong file .env."
                )

            client = genai.Client(api_key=api_key)

            try:
                answer = rag.generate_answer(
                    client,
                    llm_model,
                    question,
                    results,
                )
            finally:
                client.close()

        assistant_message = {
            "role": "assistant",
            "content": answer,
            "sources": results,
            "elapsed": perf_counter() - started,
        }

    except Exception as error:
        assistant_message = {
            "role": "assistant",
            "content": f"Không tạo được câu trả lời: {error}",
            "error": True,
            "sources": results,
            "elapsed": perf_counter() - started,
        }

    st.session_state.messages.append(assistant_message)
    show_message(assistant_message)


def knowledge_page(index, chunks):
    st.title("Kho tài liệu")

    documents = {}

    for chunk in chunks:
        document_id = chunk["document_id"]

        if document_id not in documents:
            documents[document_id] = {
                "title": chunk.get("title", ""),
                "chunks": [],
            }

        documents[document_id]["chunks"].append(chunk)

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Tài liệu", len(documents))
    col2.metric("Chunks", len(chunks))
    col3.metric("Vectors", index.ntotal)
    col4.metric("Số chiều vector", index.d)

    st.caption(
        f"Embedding: {rag.EXPECTED_MODEL}"
    )

    document_ids = sorted(documents)

    selected_id = st.selectbox(
        "Chọn tài liệu",
        options=document_ids,
        format_func=lambda value: (
            documents[value]["title"] or value
        ),
    )

    selected_chunks = documents[selected_id]["chunks"]

    st.dataframe(
        [
            {
                "Chunk ID": chunk.get("chunk_id", ""),
                "Mục": chunk.get("heading", ""),
                "Số từ": len(chunk.get("text", "").split()),
            }
            for chunk in selected_chunks
        ],
        use_container_width=True,
        hide_index=True,
    )

    selected_position = st.selectbox(
        "Chọn chunk để xem chi tiết",
        options=list(range(len(selected_chunks))),
        format_func=lambda position: selected_chunks[
            position
        ]["chunk_id"],
    )

    chunk = selected_chunks[selected_position]

    st.subheader(chunk.get("heading") or "Nội dung chunk")

    show_source_url(chunk.get("url", ""))
    st.write(chunk.get("text", ""))

    with st.expander("Metadata của chunk"):
        st.json(
            {
                key: value
                for key, value in chunk.items()
                if key != "text"
            }
        )


def main():
    st.session_state.setdefault("messages", [])

    try:
        with st.spinner("Đang tải model và kho vector..."):
            index, chunks, model, lock = load_resources(
                store_version()
            )
    except Exception as error:
        st.error(str(error))
        st.code("python build_index.py", language="bash")
        st.stop()

    with st.sidebar:
        st.title("Diabetes RAG")

        page = st.radio(
            "Chức năng",
            ["Hỏi đáp", "Kho tài liệu"],
        )

        top_k = st.slider(
            "Số đoạn dùng để trả lời",
            min_value=1,
            max_value=rag.TOP_K,
            value=rag.TOP_K,
        )

        st.caption(
            f"Mô hình sinh: "
            f"{os.getenv('GEMINI_MODEL', 'Chưa cấu hình')}"
        )

        if st.button(
            "Xóa hội thoại",
            use_container_width=True,
        ):
            st.session_state.messages = []
            st.rerun()

    if page == "Hỏi đáp":
        chat_page(index, chunks, model, lock, top_k)
    else:
        knowledge_page(index, chunks)


if __name__ == "__main__":
    main() 

#python -m streamlit run app.py