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

    client.post("/api/ask", json={"query":"What is the refund window for small appliances?"})
    r2 = client.get("/api/metrics")
    assert r2.status_code == 200
    data2 = r2.json()
    assert data2["total_docs"] == data["total_docs"]
    assert data2["total_chunks"] == data["total_chunks"]
    assert data2["avg_retrieval_latency_ms"] > 0
    assert data2["avg_generation_latency_ms"] > 0

def test_ingest_and_ask(client):
    r = client.post("/api/ingest")
    assert r.status_code == 200
    data = r.json()
    assert "indexed_docs" in data
    assert "indexed_chunks" in data

    # Ask a deterministic question
    r2 = client.post("/api/ask", json={"query":"What is the refund window for small appliances?"})
    assert r2.status_code == 200
    data2 = r2.json()
    assert "query" in data2 and data2["query"] == "What is the refund window for small appliances?"
    assert "citations" in data2 and len(data2["citations"]) > 0
    assert "chunks" in data2 and len(data2["chunks"]) > 0
    assert "answer" in data2 and isinstance(data2["answer"], str)
    assert "metrics" in data2 and "retrieval_ms" in data2["metrics"] and "generation_ms" in data2["metrics"]

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

def test_acceptance_ques_refund_blender_20_days_has_expected_sources(client):
    client.post("/api/ingest")
    r = client.post("/api/ask", json={"query":"Can a customer return a damaged blender after 20 days?"})
    assert r.status_code == 200
    data = r.json()
    # citations include Returns_and_Refunds.md and Warranty_Policy.md
    assert "citations" in data and any("Returns_and_Refunds.md" in c["title"] for c in data["citations"]) and any("Warranty_Policy.md" in c["title"] for c in data["citations"])

def test_acceptance_ques_shipping_sla_east_malaysia_bulky_has_expected_source(client):
    client.post("/api/ingest")
    r = client.post("/api/ask", json={"query":"What is the shipping SLA for bulky items in East Malaysia?"})
    assert r.status_code == 200
    data = r.json()
    # citations include Delivery_and_Shipping.md
    assert "citations" in data and any("Delivery_and_Shipping.md" in c["title"] for c in data["citations"])
