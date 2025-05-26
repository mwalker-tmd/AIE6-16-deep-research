# src/open_deep_research/cache.py

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
import sqlite3
import hashlib
import numpy as np

class HybridCache:
    def __init__(self, db_path="cache.sqlite", qdrant_url="http://localhost:6333", collection_name="deep_cache", embed_fn=None):
        self.conn = sqlite3.connect(db_path)
        self.conn.execute("CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, value TEXT)")
        self.qdrant = QdrantClient(url=qdrant_url)
        self.collection = collection_name
        self.embed = embed_fn
        self.qdrant.recreate_collection(
            collection_name=self.collection,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE)  # Update size for your model
        )

    def _hash(self, text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()

    def get_exact(self, text: str) -> str | None:
        key = self._hash(text)
        row = self.conn.execute("SELECT value FROM cache WHERE key = ?", (key,)).fetchone()
        return row[0] if row else None

    def put_exact(self, text: str, value: str):
        key = self._hash(text)
        self.conn.execute("INSERT OR REPLACE INTO cache (key, value) VALUES (?, ?)", (key, value))
        self.conn.commit()

    def get_semantic(self, query: str, threshold: float = 0.85) -> str | None:
        if not self.embed:
            return None
        vector = self.embed(query)
        hits = self.qdrant.search(
            collection_name=self.collection,
            query_vector=vector,
            limit=1
        )
        if hits and hits[0].score > threshold:
            return hits[0].payload["value"]
        return None

    def put_semantic(self, query: str, value: str):
        if not self.embed:
            return
        vector = self.embed(query)
        self.qdrant.upsert(
            collection_name=self.collection,
            points=[
                PointStruct(
                    id=self._hash(query),
                    vector=vector,
                    payload={"value": value}
                )
            ]
        )
