import json
import time
import statistics
from urllib import request, error

BASE_URL = "http://localhost:8000"
ASK_URL = BASE_URL + "/api/ask"

EVAL_SET = [
{
"query": "Can a customer return a damaged blender after 20 days?",
"expected_sources": ["Returns_and_Refunds.md", "Warranty_Policy.md"],
},
{
"query": "What's the shipping SLA to East Malaysia for bulky items?",
"expected_sources": ["Delivery_and_Shipping.md"],
},
{
"query": "Is there a surcharge for bulky deliveries?",
"expected_sources": ["Delivery_and_Shipping.md"],
},
{
"query": "How long is warranty support for appliances?",
"expected_sources": ["Warranty_Policy.md"],
},
{
"query": "Can I return an opened book if I don't like it?",
"expected_sources": ["Returns_and_Refunds.md"],
}
]

def post_json(url, payload):
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(
    url,
    data=data,
    headers={"Content-Type": "application/json"},
    method="POST",
    )
    with request.urlopen(req, timeout=30) as resp:
        body = resp.read().decode("utf-8")
    return json.loads(body)

def p95(values):
    if not values:
        return 0.0
    s = sorted(values)
    idx = int(0.95 * (len(s) - 1))
    return s[idx]

def run_eval():
    results = []
    latencies = []
    hit_count = 0
    top1_count = 0

    for item in EVAL_SET:
        q = item["query"]
        expected = set(item["expected_sources"])

        t0 = time.perf_counter()
        try:
            resp = post_json(ASK_URL, {"query": q, "k": 4})
        except error.URLError as e:
            results.append({
                "query": q,
                "ok": False,
                "error": str(e),
            })
            continue
        dt_ms = (time.perf_counter() - t0) * 1000.0
        latencies.append(dt_ms)

        citations = resp.get("citations", [])
        titles = [c.get("title", "") for c in citations]

        hit = any(t in expected for t in titles)
        top1_hit = len(titles) > 0 and titles[0] in expected

        if hit:
            hit_count += 1
        if top1_hit:
            top1_count += 1

        results.append({
            "query": q,
            "ok": True,
            "latency_ms": round(dt_ms, 2),
            "expected": sorted(expected),
            "returned": titles,
            "hit": hit,
            "top1_hit": top1_hit,
        })

    total = len(EVAL_SET)
    summary = {
        "total": total,
        "answered": len([r for r in results if r.get("ok")]),
        "citation_hit_rate": round(hit_count / total, 3) if total else 0.0,
        "top1_hit_rate": round(top1_count / total, 3) if total else 0.0,
        "avg_latency_ms": round(statistics.mean(latencies), 2) if latencies else 0.0,
        "p95_latency_ms": round(p95(latencies), 2) if latencies else 0.0,
    }

    print("=== EVAL SUMMARY ===")
    print(json.dumps(summary, indent=2))
    print("\n=== PER QUERY ===")
    for r in results:
        print(json.dumps(r, indent=2))

if __name__ == "__main__":
    run_eval()