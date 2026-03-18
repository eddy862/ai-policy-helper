import numpy as np
from app.rag import (
    LocalEmbedder,
    InMemoryStore,
    _tokenize,
    _build_lexical_meta,
    RAGEngine,
)


def test_local_embedder_deterministic():
    """Same text always produces same vector."""
    embedder = LocalEmbedder(dim=384)
    text = "refund policy for customers"
    
    v1 = embedder.embed(text)
    v2 = embedder.embed(text)
    
    assert np.allclose(v1, v2), "Same text should produce identical vectors"
    assert abs(np.linalg.norm(v1) - 1.0) < 1e-5, "Vector should be L2 normalized"


def test_local_embedder_different_texts():
    """Different texts produce different vectors."""
    embedder = LocalEmbedder(dim=384)
    v1 = embedder.embed("refund policy")
    v2 = embedder.embed("warranty information")
    
    assert not np.allclose(v1, v2), "Different texts should produce different vectors"


def test_tokenize_strips_and_lowercases():
    """Tokenizer handles case and special chars."""
    result = _tokenize("Hello! World's best POLICY.")
    assert result == ["hello", "world's", "best", "policy"]


def test_build_lexical_meta_counts_tokens():
    """Lexical meta builds TF and length correctly."""
    text = "the quick brown fox jumps over the lazy dog"
    meta = _build_lexical_meta(text)
    
    assert meta["length"] == 9
    assert meta["tf"]["the"] == 2
    assert meta["tf"]["quick"] == 1
    assert len(meta["tokens"]) == 9


def test_in_memory_store_upsert_and_search():
    """Store inserts and retrieves vectors by similarity."""
    store = InMemoryStore(dim=3)
    
    # Simple vectors: e1 = [1, 0, 0], e2 = [0, 1, 0], query = [1, 0, 0]
    v1 = np.array([1.0, 0.0, 0.0], dtype="float32")
    v2 = np.array([0.0, 1.0, 0.0], dtype="float32")
    
    # vector + meta
    store.upsert(
        [v1, v2],
        [
            {"hash": "h1", "title": "doc1", "text": "text1"},
            {"hash": "h2", "title": "doc2", "text": "text2"},
        ],
    )
    
    query = np.array([1.0, 0.0, 0.0], dtype="float32")
    results = store.search(query, k=2)
    
    assert len(results) == 2
    assert results[0][1]["title"] == "doc1"  # v1 should rank first
    assert results[1][1]["title"] == "doc2"


def test_in_memory_store_deduplication_by_hash():
    """Store skips duplicate hashes."""
    store = InMemoryStore(dim=3)
    v1 = np.array([1.0, 0.0, 0.0], dtype="float32")
    
    store.upsert(
        [v1],
        [{"hash": "h1", "title": "doc1", "text": "text1"}],
    )
    
    initial_count = len(store.vecs)
    
    # Try to upsert same hash again
    store.upsert(
        [v1],
        [{"hash": "h1", "title": "doc1", "text": "text1"}],
    )
    
    assert len(store.vecs) == initial_count, "Duplicate hash should not be added"


def test_rag_engine_retrieve_returns_scored_results():
    """Retrieve returns results with scores up to k."""
    engine = RAGEngine()
    
    # Manually ingest some simple chunks
    chunks = [
        {"title": "policy.md", "section": "Returns", "text": "refund window is 30 days"},
        {"title": "policy.md", "section": "Warranty", "text": "warranty covers manufacturing defects"},
    ]
    
    engine.ingest_chunks(chunks)
    
    results = engine.retrieve("refund", k=2)
    
    assert len(results) <= 2
    assert all("score" in r for r in results)
    assert all("title" in r for r in results)
    assert all("text" in r for r in results)


def test_rag_engine_retrieve_returns_empty_when_no_docs():
    """Retrieve gracefully returns empty list when store is empty."""
    engine = RAGEngine()
    results = engine.retrieve("any query", k=4)
    
    assert results == []


def test_rrf_fuse_combines_rankings():
    """RRF properly combines dense and lexical rankings."""
    engine = RAGEngine()
    
    dense_results = [
        (0.9, {"hash": "h1", "title": "doc1", "text": "text1"}),
        (0.7, {"hash": "h2", "title": "doc2", "text": "text2"}), # h2 ranks second in dense
    ]
    
    lexical_results = [
        (5.0, {"hash": "h2", "title": "doc2", "text": "text2"}),  # h2 ranks first in lexical
        (3.0, {"hash": "h3", "title": "doc3", "text": "text3"}),
    ]
    
    fused = engine._rrf_fuse(dense_results, lexical_results, final_k=2, fusion_k=60)
    
    assert len(fused) == 2
    for score, meta in fused:
        assert isinstance(score, float), "Score should be RRF float value"
        assert isinstance(meta, dict), "Metadata should be a dict"
        assert "hash" in meta and "title" in meta and "text" in meta, "Metadata should contain hash, title, and text"
        
    # h2 should rank high because it appears in both rankings
    assert fused[0][1]["hash"] == "h2", "Fused result should prioritize items in both rankings"