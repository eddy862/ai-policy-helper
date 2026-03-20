# integration tests for the API endpoints
def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

# this should be run first since the scope is session - it will ingest the docs and populate the vector store for subsequent tests
def test_metrics_after_ingest_and_ask(client):
    client.post("/api/ingest")
    r = client.get("/api/metrics")
    assert r.status_code == 200
    data = r.json()
    assert data["total_docs"] > 0
    assert data["total_chunks"] > 0
    assert data["avg_retrieval_latency_ms"] == 0
    assert data["avg_generation_latency_ms"] == 0

    ask_r = client.post("/api/ask", json={"query":"What is the refund window for small appliances?"})
    assert ask_r.status_code == 200
    ask_data = ask_r.json()
    r2 = client.get("/api/metrics")
    assert r2.status_code == 200
    data2 = r2.json()
    assert data2["total_docs"] == data["total_docs"]
    assert data2["total_chunks"] == data["total_chunks"]
    assert data2["avg_retrieval_latency_ms"] > 0

    if ask_data["metrics"].get("needs_clarification"):
        assert data2["avg_generation_latency_ms"] == 0
    else:
        assert data2["avg_generation_latency_ms"] > 0

def test_ask(client):
    # Ask a deterministic question
    r = client.post("/api/ask", json={"query":"What is the refund window for small appliances?"})
    assert r.status_code == 200
    data = r.json()
    assert "query" in data and data["query"] == "What is the refund window for small appliances?"
    assert "citations" in data and len(data["citations"]) > 0
    assert "chunks" in data and len(data["chunks"]) > 0
    assert "answer" in data and isinstance(data["answer"], str)
    assert "metrics" in data and "retrieval_ms" in data["metrics"] and "generation_ms" in data["metrics"] and "confidence" in data["metrics"] and "needs_clarification" in data["metrics"] and "confidence_reason" in data["metrics"] and "top_score" in data["metrics"] and "score_gap" in data["metrics"] and "source_diversity" in data["metrics"]

def test_ask_low_confidence(client):
    r = client.post("/api/ask", json={"query":"Hi"})
    assert r.status_code == 200
    data = r.json()
    assert data["metrics"]["needs_clarification"] == True
    assert "I found related info across" in data["answer"]

def test_ask_high_confidence(client):
    r = client.post("/api/ask", json={"query":"Can a customer return a damaged blender after 20 days?"})
    assert r.status_code == 200
    data = r.json()
    assert data["metrics"]["needs_clarification"] == False
    assert "I found related info across" not in data["answer"]

def test_ask_missing_query(client):
    r = client.post("/api/ask", json={})
    assert r.status_code == 422  

def test_ask_wrong_type_query(client):
    r = client.post("/api/ask", json={"query": 123})
    assert r.status_code == 422  

    r2 = client.post("/api/ask", json={"query": "What is the refund window?", "k": "five"})
    assert r2.status_code == 422  

    r3 = client.post("/api/ask", json={"query": "What is the refund window?", "dense_k": "five"})
    assert r3.status_code == 422 

    r4 = client.post("/api/ask", json={"query": "What is the refund window?", "lexical_k": "five"})
    assert r4.status_code == 422  

def test_metrics(client):
    r = client.get("/api/metrics")
    assert r.status_code == 200
    data = r.json()
    assert "total_docs" in data
    assert "total_chunks" in data
    assert "avg_retrieval_latency_ms" in data
    assert "avg_generation_latency_ms" in data
    assert "embedding_model" in data
    assert "llm_model" in data

