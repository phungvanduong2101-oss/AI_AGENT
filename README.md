# AI_AGENT · Knowledge Graph và RAG về đái tháo đường

Ứng dụng tra cứu tài liệu bằng Streamlit. Mỗi câu hỏi được mã hóa bằng
`intfloat/multilingual-e5-small`. FAISS tìm tối đa 15 đoạn gần nhất. Neo4j
tìm các đoạn trong nhóm đó có liên kết với chủ đề được nhắc đến trong
câu hỏi. Đoạn có chủ đề khớp được cộng 0,04 điểm khi xếp hạng lại. Tối đa
3 đoạn được đưa vào Gemini để tạo câu trả lời tiếng Việt kèm số trích
dẫn. Mục nguồn trong Streamlit hiện văn bản, URL và chủ đề Neo4j đã khớp.

Prototype phục vụ học tập và tra cứu. Câu trả lời cần được đối chiếu với
tài liệu nguồn. Ứng dụng không chẩn đoán cá nhân hay kê thuốc.

## 1. Chuẩn bị trên Windows

Cài Python 3.12 và Docker Desktop, sau đó bật Docker Desktop. Mở PowerShell
trong thư mục `AI_AGENT` sau khi giải nén.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
notepad .env
```

Điền vào `.env`:

| Biến | Giá trị |
|---|---|
| `GEMINI_API_KEY` | API key từ [Google AI Studio](https://aistudio.google.com/apikey). |
| `GEMINI_MODEL` | Mặc định là `gemini-2.5-flash`. Chọn model có quyền truy cập theo [danh sách Gemini](https://ai.google.dev/gemini-api/docs/models). |
| `NEO4J_URI` | `bolt://localhost:7687`. |
| `NEO4J_USER` | `neo4j` cho cấu hình Docker này. |
| `NEO4J_PASSWORD` | Mật khẩu do bạn đặt, từ 8 ký tự trở lên. Có thể dùng chữ và số để tránh lỗi do ký tự đặc biệt trong lệnh shell. |
| `NEO4J_DATABASE` | `neo4j`. |

Trên macOS/Linux dùng `python3 -m venv .venv`, `source .venv/bin/activate`
và thay `.\.venv\Scripts\python.exe` bằng `python`.

## 2. Tạo CSV và FAISS index

```powershell
.\.venv\Scripts\python.exe chunk_documents.py
.\.venv\Scripts\python.exe export_kg_csv.py
.\.venv\Scripts\python.exe build_index.py
```

Lần đầu tạo index cần mạng để tải model embedding từ Hugging Face.
`data/documents/` chứa JSON tài liệu nguồn. `chunk_documents.py` chia các
mục thành đoạn có ID và lưu vào `data/chunks.json`. `export_kg_csv.py` dùng
cùng danh sách đoạn để sinh CSV cho Neo4j. `build_index.py` tạo
`data/vector_store/` cho FAISS.

| CSV trong `neo4j/import/` | Nội dung |
|---|---|
| `documents.csv` | ID, tiêu đề, URL của 25 tài liệu. |
| `chunks.csv` | 475 đoạn, ID tài liệu, tiêu đề mục và nội dung. |
| `topics.csv` | 10 chủ đề và cụm từ nhận diện, có thể chỉnh tay. |
| `chunk_topics.csv` | Liên kết đoạn–chủ đề, tạo từ tiêu đề tài liệu và tiêu đề mục. |

`data/diabetes_prediction_dataset.csv` thuộc dữ liệu cho bài toán dự đoán.
Knowledge Graph dùng CSV xuất từ kho tài liệu. Gắn chủ đề theo tiêu đề
là một phương pháp lập chỉ mục. Quan hệ `ABOUT` thể hiện nhãn nội dung,
không biểu diễn quan hệ nhân quả hay kết luận y khoa.

## 3. Import Knowledge Graph bằng Cypher

### Cách A · Docker Desktop

```powershell
docker compose up -d
docker compose logs neo4j --tail 30
```

Đợi log báo `Started.`, rồi chạy lệnh import. Khi shell hỏi mật khẩu,
nhập `NEO4J_PASSWORD` trong `.env`.

```powershell
docker compose exec neo4j cypher-shell -u neo4j -d neo4j -f /project/DB_diabet.cypher
```

Mở [Neo4j Browser](http://localhost:7474), đăng nhập bằng `neo4j` và mật
khẩu đã cấu hình. Khi dùng lại volume cũ, Neo4j giữ mật khẩu của lần khởi
tạo trước.

### Cách B · Neo4j Desktop 5.26

Tạo và khởi động một instance Neo4j 5.26. Mở thư mục `import` của instance
và chép cả bốn file CSV từ `neo4j/import/` vào đó. Chạy từng câu trong
`neo4j/DB_diabet.cypher` bằng Neo4j Browser, hoặc dùng `cypher-shell` với
tham số `-f` trỏ đến file. Cập nhật URI, mật khẩu và tên database trong
`.env` theo instance. Đường dẫn `file:///` được Neo4j giải từ thư mục import
trên máy chạy database.

### Kiểm tra graph

Chạy từng câu trong `neo4j/queries.cypher`. Ví dụ để xem graph HbA1c:

```cypher
MATCH p = (:RagDocument)-[:HAS_CHUNK]->(:RagChunk)-[:ABOUT]->(:RagTopic {id: 'hba1c'})
RETURN p LIMIT 20;
```

Script tạo ràng buộc duy nhất cho ID và dùng `MERGE`, nên có thể chạy
lại với cùng bộ CSV. Khi xóa tài liệu hoặc đổi từ khóa chủ đề, hãy tạo lại
CSV và index. Trên database dành riêng cho đồ án, chạy ba câu dưới để
xóa projection cũ trước khi import lại. Các câu này xóa cả quan hệ nối
với node được chọn, vì vậy chỉ chạy khi muốn dựng lại graph của dự án.

```cypher
MATCH (c:RagChunk) DETACH DELETE c;
MATCH (d:RagDocument) DETACH DELETE d;
MATCH (t:RagTopic) DETACH DELETE t;
```

## 4. Chạy và demo

```powershell
.\.venv\Scripts\python.exe search.py
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Gõ `exit` để thoát script tìm kiếm trước khi chạy Streamlit. `search.py`
dùng cùng luồng truy xuất với giao diện và không gọi Gemini. Mở
[Streamlit](http://localhost:8501) và thử câu **“Chỉ số HbA1c được giải thích
thế nào trong tài liệu?”**. Mở mục nguồn dưới câu trả lời để đối chiếu
đoạn trích. Có thể chạy `rag.py` để dùng RAG từ terminal với Gemini.

Để chạy riêng chế độ FAISS, đặt `NEO4J_URI=` trong `.env`. Khi đã cấu hình
URI và câu hỏi có chủ đề nhưng Neo4j không kết nối được, ứng dụng báo lỗi
để dễ phát hiện cấu hình sai. Nếu truy xuất có kết quả mà không có chủ đề
Neo4j, kiểm tra graph đã import và hỏi bằng từ khóa trong `topics.csv`.

## 5. Khi đẩy lên GitHub

Giữ mã nguồn, `.env.example`, các script Cypher và cấu hình Docker.
`.gitignore` đã loại `.env`, cache Python, `data/chunks.json` và FAISS
index. CSV Knowledge Graph có kèm trong bản ZIP để chạy thử. Trước khi
đăng công khai tài liệu hoặc CSV chứa nội dung trích từ tài liệu, kiểm
tra quyền chia sẻ của dữ liệu nguồn.
