# src/open_deep_research/cache.py

import json
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
import sqlite3
import hashlib
import numpy as np

class HybridCache:
    def __init__(self, db_path="cache.sqlite", qdrant_url="http://localhost:6333", collection_name="deep_cache", embed_fn=None, timeout=30, ttl=3600):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.execute("CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, value TEXT)")
        self.qdrant = QdrantClient(url=qdrant_url, timeout=timeout)
        self.collection = collection_name
        self.embed = embed_fn
        self.ttl = ttl
        self.qdrant.recreate_collection(
            collection_name=self.collection,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE)  # Update size for your model
        )

    def _hash(self, text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()

    def _serialize_value(self, value):
        """Serialize value to JSON string."""
        return json.dumps(value)

    def _deserialize_value(self, value_str):
        """Deserialize JSON string to Python object."""
        return json.loads(value_str) if value_str else None

    def get_exact(self, text: str) -> str | None:
        key = self._hash(text)
        row = self.conn.execute("SELECT value FROM cache WHERE key = ?", (key,)).fetchone()
        return self._deserialize_value(row[0]) if row else None

    def put_exact(self, text: str, value):
        key = self._hash(text)
        serialized_value = self._serialize_value(value)
        self.conn.execute("INSERT OR REPLACE INTO cache (key, value) VALUES (?, ?)", (key, serialized_value))
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
            return self._deserialize_value(hits[0].payload["value"])
        return None

    def put_semantic(self, query: str, value):
        if not self.embed:
            return
        vector = self.embed(query)
        serialized_value = self._serialize_value(value)
        self.qdrant.upsert(
            collection_name=self.collection,
            points=[
                PointStruct(
                    id=self._hash(query),
                    vector=vector,
                    payload={"value": serialized_value}
                )
            ]
        )

    # Alias methods for backward compatibility
    def set(self, text: str, value):
        """Alias for put_exact."""
        self.put_exact(text, value)

    def get(self, text: str) -> str | None:
        """Alias for get_exact."""
        return self.get_exact(text)
