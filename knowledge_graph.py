import os
from functools import lru_cache

from neo4j import GraphDatabase

from kg_topics import load_topics, match_topics

TOPIC_QUERY = """
MATCH (d:RagDocument)-[:HAS_CHUNK]->(c:RagChunk)-[:ABOUT]->(t:RagTopic)
WHERE c.id IN $chunk_ids AND t.id IN $topic_ids
RETURN c.id AS chunk_id, collect(DISTINCT t.name) AS topics
"""


@lru_cache(maxsize=1)
def topic_vocabulary():
    return load_topics()


def find_topics(question, chunk_ids):
    uri = os.getenv("NEO4J_URI", "").strip()
    topic_ids = match_topics(question, topic_vocabulary())
    # Chủ đề chung xuất hiện trong nhiều tài liệu, không dùng để ưu tiên.
    topic_ids = [topic for topic in topic_ids if topic != "diabetes"]
    if not uri or not topic_ids or not chunk_ids:
        return {}

    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "")
    database = os.getenv("NEO4J_DATABASE", "neo4j")
    with GraphDatabase.driver(uri, auth=(user, password), connection_timeout=5) as driver:
        with driver.session(database=database) as session:
            rows = session.run(TOPIC_QUERY, chunk_ids=chunk_ids, topic_ids=topic_ids)
            return {row["chunk_id"]: sorted(row["topics"]) for row in rows}
