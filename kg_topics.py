import csv
import re
import unicodedata
from pathlib import Path

TOPICS_FILE = Path(__file__).resolve().parent / "neo4j/import/topics.csv"


def normalize(text):
    text = unicodedata.normalize("NFC", text.casefold())
    return " " + " ".join(re.findall(r"\w+", text)) + " "


def load_topics():
    with TOPICS_FILE.open(encoding="utf-8", newline="") as source:
        return {
            row["topic_id"]: tuple(normalize(alias) for alias in row["aliases"].split("|"))
            for row in csv.DictReader(source)
        }


def match_topics(text, topics):
    normalized = normalize(text)
    return [topic_id for topic_id, aliases in topics.items()
            if any(alias in normalized for alias in aliases)]
