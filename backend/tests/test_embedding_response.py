"""`/embedding` response parsing — llama-server's shape has changed across
releases, and the mismatch only surfaces at the first long-term memory
search (a live server is needed to see it), so pin every shape here."""
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.memory.embedder import _extract_embedding


def test_legacy_flat_object():
    assert _extract_embedding({"embedding": [0.1, 0.2]}) == [0.1, 0.2]


def test_current_list_of_nested_rows():
    """b10016's actual shape, captured from a running server."""
    payload = [{"index": 0, "embedding": [[0.1, 0.2, 0.3]]}]
    assert _extract_embedding(payload) == [0.1, 0.2, 0.3]


def test_list_without_nesting():
    assert _extract_embedding([{"index": 0, "embedding": [0.1, 0.2]}]) == [0.1, 0.2]


def test_openai_style_data_wrapper():
    assert _extract_embedding({"data": [{"embedding": [0.5]}]}) == [0.5]


def test_integers_are_coerced_to_float():
    vector = _extract_embedding({"embedding": [1, 2]})
    assert vector == [1.0, 2.0]
    assert all(isinstance(value, float) for value in vector)


@pytest.mark.parametrize("payload", [[], {}, {"embedding": []}, "nope", {"embedding": [[]]}])
def test_unusable_payloads_raise(payload):
    with pytest.raises(ValueError):
        _extract_embedding(payload)
