"""Shared Chroma persistent client — one collection, one process-wide
instance, matching the single-local-user model everywhere else in this app."""
from app.config import VECTOR_STORE_DIR

_client = None
_collection = None


def get_collection():
    global _client, _collection
    if _collection is None:
        import chromadb  # deferred: not needed until memory is actually touched

        VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(path=str(VECTOR_STORE_DIR))
        _collection = _client.get_or_create_collection("long_term_memory")
    return _collection
