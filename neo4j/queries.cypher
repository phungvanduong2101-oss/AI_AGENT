// Đếm số tài liệu, đoạn, chủ đề và liên kết.
MATCH (d:RagDocument) RETURN count(d) AS documents;
MATCH (c:RagChunk) RETURN count(c) AS chunks;
MATCH (t:RagTopic) RETURN count(t) AS topics;
MATCH (:RagChunk)-[r:ABOUT]->(:RagTopic) RETURN count(r) AS topic_links;

// Hiển thị graph nguồn cho chủ đề HbA1c.
MATCH p = (:RagDocument)-[:HAS_CHUNK]->(:RagChunk)-[:ABOUT]->(:RagTopic {id: 'hba1c'})
RETURN p LIMIT 20;
