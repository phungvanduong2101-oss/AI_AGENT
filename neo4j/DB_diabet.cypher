CREATE CONSTRAINT rag_document_id IF NOT EXISTS FOR (d:RagDocument) REQUIRE d.id IS UNIQUE;
CREATE CONSTRAINT rag_chunk_id IF NOT EXISTS FOR (c:RagChunk) REQUIRE c.id IS UNIQUE;
CREATE CONSTRAINT rag_topic_id IF NOT EXISTS FOR (t:RagTopic) REQUIRE t.id IS UNIQUE;

LOAD CSV WITH HEADERS FROM 'file:///topics.csv' AS row
MERGE (t:RagTopic {id: row.topic_id})
SET t.name = row.name, t.aliases = split(row.aliases, '|');

LOAD CSV WITH HEADERS FROM 'file:///documents.csv' AS row
MERGE (d:RagDocument {id: row.document_id})
SET d.title = row.title, d.url = row.url;

LOAD CSV WITH HEADERS FROM 'file:///chunks.csv' AS row
MATCH (d:RagDocument {id: row.document_id})
MERGE (c:RagChunk {id: row.chunk_id})
SET c.heading = coalesce(row.heading, ''), c.text = row.text,
    c.section_index = toInteger(row.section_index)
MERGE (d)-[:HAS_CHUNK]->(c);

LOAD CSV WITH HEADERS FROM 'file:///chunk_topics.csv' AS row
MATCH (c:RagChunk {id: row.chunk_id}), (t:RagTopic {id: row.topic_id})
MERGE (c)-[:ABOUT]->(t);
