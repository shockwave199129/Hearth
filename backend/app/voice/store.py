"""Voiceprint persistence.

**A stored voiceprint is biometric data.** It is a template derived from a
person's voice and used to identify them, which puts it under Illinois BIPA,
Texas CUBI, Washington's biometric law, and GDPR Art. 9 as special-category
data — see docs/compliance.md. Consequences that are this module's job:

- It is Fernet-encrypted at rest like transcript content, not stored raw.
- It never leaves the device, and it is excluded from crash reports and from
  the plaintext data export (`app.data_export` exports the *fact* that one
  exists, never the vector).
- It is deletable on its own, without deleting the profile, and it is purged
  by profile deletion — see the cascade in `app/api/profile.py`.

``model_id`` is stored alongside the vector because a template is only
meaningful to the model that produced it. Comparing a stored centroid
against an embedding from a different model yields a plausible-looking
number with no meaning, so a model change must invalidate enrollment rather
than silently degrade it.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np

from app.config import DATA_DIR
from app.db.sqlite_models import get_connection
from app.security.crypto import decrypt, encrypt

logger = logging.getLogger("hearth.speaker")

VOICEPRINT_DB_PATH = DATA_DIR / "profile.db"


@dataclass(frozen=True)
class Voiceprint:
    user_id: str
    embedding: np.ndarray
    sample_count: int
    model_id: str
    enrolled_at: str
    updated_at: str


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def save(user_id: str, embedding: np.ndarray, *, sample_count: int, model_id: str) -> Voiceprint:
    """Insert or replace this profile's voiceprint. One per profile."""
    vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
    payload = encrypt(json.dumps([float(x) for x in vector])).decode("latin1")
    now = _now_iso()
    conn = get_connection(VOICEPRINT_DB_PATH)
    try:
        existing = conn.execute(
            "SELECT enrolled_at FROM voiceprints WHERE user_id = ?", (user_id,)
        ).fetchone()
        enrolled_at = existing[0] if existing else now
        conn.execute(
            """INSERT INTO voiceprints
                   (user_id, embedding, sample_count, dim, model_id, enrolled_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(user_id) DO UPDATE SET
                   embedding = excluded.embedding,
                   sample_count = excluded.sample_count,
                   dim = excluded.dim,
                   model_id = excluded.model_id,
                   updated_at = excluded.updated_at""",
            (user_id, payload, sample_count, len(vector), model_id, enrolled_at, now),
        )
        conn.commit()
    finally:
        conn.close()
    return Voiceprint(user_id, vector, sample_count, model_id, enrolled_at, now)


def get(user_id: str) -> Voiceprint | None:
    conn = get_connection(VOICEPRINT_DB_PATH)
    try:
        row = conn.execute(
            """SELECT embedding, sample_count, model_id, enrolled_at, updated_at
               FROM voiceprints WHERE user_id = ?""",
            (user_id,),
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        return None
    try:
        vector = np.asarray(json.loads(decrypt(row[0].encode("latin1"))), dtype=np.float32)
    except Exception:
        # An unreadable template is treated as absent rather than raising:
        # the consequence is "not enrolled", which is a safe state, whereas
        # raising would take out every voice turn.
        logger.exception("voiceprint for %s could not be decrypted — treating as absent", user_id)
        return None
    return Voiceprint(user_id, vector, int(row[1]), str(row[2]), str(row[3]), str(row[4]))


def delete(user_id: str) -> None:
    """Standalone 'forget my voice'. Also called by the profile-delete cascade."""
    conn = get_connection(VOICEPRINT_DB_PATH)
    try:
        conn.execute("DELETE FROM voiceprints WHERE user_id = ?", (user_id,))
        conn.commit()
    finally:
        conn.close()


def metadata(user_id: str) -> dict:
    """Everything about the enrollment *except* the template itself.

    This is what the API and the data export are allowed to show. Exporting
    or returning the vector would put a biometric template in plaintext.
    """
    print_ = get(user_id)
    if print_ is None:
        return {"enrolled": False}
    return {
        "enrolled": True,
        "sample_count": print_.sample_count,
        "model_id": print_.model_id,
        "enrolled_at": print_.enrolled_at,
        "updated_at": print_.updated_at,
        "dimensions": int(print_.embedding.size),
    }
