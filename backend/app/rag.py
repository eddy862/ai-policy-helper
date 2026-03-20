import time, os, math, json, hashlib, uuid, re
from typing import List, Dict, Tuple
import numpy as np
from .settings import settings
from .ingest import chunk_text, doc_hash
from qdrant_client import QdrantClient, models as qm
from collections import Counter
import logging

logger = logging.getLogger(__name__)

# ---- Simple local embedder (deterministic) ----
def _tokenize(s: str) -> List[str]:
    # Keep alphanumerics and apostrophes, lowercase for lexical matching
    return re.findall(r"[a-z0-9']+", s.lower())

def _build_lexical_meta(text: str) -> Dict:
    tokens = _tokenize(text)
    tf_counter = Counter(tokens)
    return {
        "tokens": tokens, # List[str]
        "tf": {k: int(v) for k, v in tf_counter.items()}, # Dict[str, int]
        "length": len(tokens), # int
    }

class LocalEmbedder:
    def __init__(self, dim: int = 384):
        self.dim = dim

    def embed(self, text: str) -> np.ndarray:
        # Hash-based repeatable pseudo-embedding
        h = hashlib.sha1(text.encode("utf-8")).digest()
        rng_seed = int.from_bytes(h[:8], "big") % (2**32-1)
        rng = np.random.default_rng(rng_seed)
        v = rng.standard_normal(self.dim).astype("float32")
        # L2 normalize
        v = v / (np.linalg.norm(v) + 1e-9)
        return v

# ---- Vector store abstraction ----
class InMemoryStore:
    def __init__(self, dim: int = 384):
        self.dim = dim
        self.vecs: List[np.ndarray] = []
        self.meta: List[Dict] = []
        self._hashes = set()

    def upsert(self, vectors: List[np.ndarray], metadatas: List[Dict]):
        for v, m in zip(vectors, metadatas):
            h = m.get("hash")
            if h and h in self._hashes:
                continue
            self.vecs.append(v.astype("float32"))
            self.meta.append(m)
            if h:
                self._hashes.add(h)

    def search(self, query: np.ndarray, k: int = 4) -> List[Tuple[float, Dict]]:
        if not self.vecs:
            return []
        A = np.vstack(self.vecs)  # [N, d]
        q = query.reshape(1, -1)  # [1, d]
        # cosine similarity
        sims = (A @ q.T).ravel() / (np.linalg.norm(A, axis=1) * (np.linalg.norm(q) + 1e-9) + 1e-9)
        idx = np.argsort(-sims)[:k]
        return [(float(sims[i]), self.meta[i]) for i in idx]

class QdrantStore:
    def __init__(self, collection: str, dim: int = 384):
        self.client = QdrantClient(url="http://qdrant:6333", timeout=10.0)
        self.collection = collection
        self.dim = dim
        self._ensure_collection()

    def _ensure_collection(self):
        try:
            self.client.get_collection(self.collection)
        except Exception:
            self.client.recreate_collection(
                collection_name=self.collection,
                vectors_config=qm.VectorParams(size=self.dim, distance=qm.Distance.COSINE)
            )

    def upsert(self, vectors: List[np.ndarray], metadatas: List[Dict]):
        points = []
        for i, (v, m) in enumerate(zip(vectors, metadatas)):
            raw_id = m.get("id") or m.get("hash")
            if raw_id and len(raw_id) >= 32: 
                point_id = str(uuid.UUID(raw_id[:32])) # qdrant accept standard UUID strings (36 chars with hyphens, e.g. 550e8400-e29b-41d4-a716-446655440000)
            else:
                point_id = i
            points.append(qm.PointStruct(id=point_id, vector=v.tolist(), payload=m))
        self.client.upsert(collection_name=self.collection, points=points)

    def search(self, query: np.ndarray, k: int = 4) -> List[Tuple[float, Dict]]:
        res = self.client.search(
            collection_name=self.collection,
            query_vector=query.tolist(),
            limit=k,
            with_payload=True
        )
        out = []
        for r in res:
            out.append((float(r.score), dict(r.payload)))
        return out

# ---- LLM provider ----
class StubLLM:
    def generate(self, query: str, contexts: List[Dict]) -> str:
        lines = [f"Answer (stub): Based on the following sources:"]
        for c in contexts:
            sec = c.get("section") or "Section"
            lines.append(f"- {c.get('title')} — {sec}")
        lines.append("Summary:")
        # naive summary of top contexts
        joined = " ".join([c.get("text", "") for c in contexts])
        lines.append(joined[:600] + ("..." if len(joined) > 600 else ""))
        return "\n".join(lines)

class OpenRouterLLM:
    def __init__(self, api_key: str, model: str = "openai/gpt-4o-mini"):
        from openai import OpenAI
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
        )
        self.model = model

    def generate(self, query: str, contexts: List[Dict]) -> str:
        prompt = f"You are a helpful company policy assistant. Cite sources by title and section when relevant.\nQuestion: {query}\nSources:\n"
        for c in contexts:
            prompt += f"- {c.get('title')} | {c.get('section')}\n{c.get('text')[:600]}\n---\n"
        prompt += "Write a concise, accurate answer grounded in the sources. If unsure, say so."
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role":"user","content":prompt}],
            temperature=0.1
        )
        return resp.choices[0].message.content

# ---- RAG Orchestrator & Metrics ----
class Metrics:
    def __init__(self):
        self.t_retrieval = []
        self.t_generation = []

    def add_retrieval(self, ms: float):
        self.t_retrieval.append(ms)

    def add_generation(self, ms: float):
        self.t_generation.append(ms)

    def summary(self) -> Dict:
        avg_r = sum(self.t_retrieval)/len(self.t_retrieval) if self.t_retrieval else 0.0
        avg_g = sum(self.t_generation)/len(self.t_generation) if self.t_generation else 0.0
        return {
            "avg_retrieval_latency_ms": round(avg_r, 2),
            "avg_generation_latency_ms": round(avg_g, 2),
        }

class RAGEngine:
    def __init__(self):
        self._df = Counter()
        self._num_docs = 0
        self._avg_doc_len = 0.0
        self._total_doc_len = 0
        self._all_meta: List[Dict] = []
        self.embedder = LocalEmbedder(dim=384)
        # Vector store selection
        if settings.vector_store == "qdrant":
            try:
                self.store = QdrantStore(collection=settings.collection_name, dim=384)
            except Exception:
                self.store = InMemoryStore(dim=384)
        else:
            self.store = InMemoryStore(dim=384)

        # LLM selection
        if settings.llm_provider == "openrouter" and settings.openrouter_api_key:
            try:
                self.llm = OpenRouterLLM(
                    api_key=settings.openrouter_api_key,
                    model=settings.llm_model,
                )
                self.llm_name = f"openrouter:{settings.llm_model}"
            except Exception:
                self.llm = StubLLM()
                self.llm_name = "stub"
        else:
            self.llm = StubLLM()
            self.llm_name = "stub"

        self.metrics = Metrics()
        self._doc_titles = set()
        self._chunk_count = 0

    def _tokenize_query(self, query: str) -> List[str]:
        return _tokenize(query)
    
    def _idf(self, term: str) -> float:
        # BM25-style IDF with smoothing
        n = max(1, self._num_docs)
        df = self._df.get(term, 0)
        return math.log(1.0 + (n - df + 0.5) / (df + 0.5))

    def _bm25_tf_norm(self, tf: int, doc_len: int, k1: float = 1.2, b: float = 0.75) -> float:
        if tf <= 0:
            return 0.0
        avg_len = self._avg_doc_len if self._avg_doc_len > 0 else 1.0
        denom = tf + k1 * (1.0 - b + b * (doc_len / avg_len))
        return (tf * (k1 + 1.0)) / (denom + 1e-9)

    def _lexical_search(self, query: str, k: int = 4) -> List[Tuple[float, Dict]]:
        q_terms = self._tokenize_query(query)
        if not q_terms:
            return []

        # Use unique query terms so repeated words do not dominate
        q_unique = set(q_terms)

        # Candidate corpus source
        if isinstance(self.store, InMemoryStore):
            candidates = self.store.meta
        else:
            # For Qdrant mode, this works if you keep a local copy of metadata
            # during ingest. Add: self._all_meta = [] in __init__, then
            # self._all_meta.extend(metas) at the end of ingest_chunks.
            candidates = getattr(self, "_all_meta", [])

        scored: List[Tuple[float, Dict]] = []
        for meta in candidates:
            tf_map = meta.get("tf", {}) or {}
            doc_len = int(meta.get("length", 0) or 0)
            if doc_len <= 0:
                continue

            score = 0.0
            for term in q_unique:
                tf = int(tf_map.get(term, 0) or 0)
                if tf <= 0:
                    continue
                score += self._idf(term) * self._bm25_tf_norm(tf, doc_len)

            if score > 0:
                scored.append((float(score), meta))

        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:k]

    def ingest_chunks(self, chunks: List[Dict]) -> Tuple[int, int]:
        vectors = []
        metas = []
        doc_titles_before = set(self._doc_titles)

        for ch in chunks:
            text = ch["text"]
            h = doc_hash(text)
            lex = _build_lexical_meta(text)
            meta = {
                "id": h,
                "hash": h,
                "title": ch["title"],
                "section": ch.get("section"),
                "text": text,
                "tokens": lex["tokens"],
                "tf": lex["tf"],
                "length": lex["length"],
            }
            v = self.embedder.embed(text)
            vectors.append(v)
            metas.append(meta)
            self._doc_titles.add(ch["title"])
            self._chunk_count += 1

            self._num_docs += 1
            self._total_doc_len += lex["length"]
            self._avg_doc_len = self._total_doc_len / max(1, self._num_docs)
            for term in set(lex["tokens"]): self._df[term] += 1


        self.store.upsert(vectors, metas)
        self._all_meta.extend(metas)
        return (len(self._doc_titles) - len(doc_titles_before), len(metas))
    
    # RRF score = 1/(fusion_k + dense_rank) + 1/(fusion_k + lexical_rank)
    def _rrf_fuse(
        self,
        dense_results: List[Tuple[float, Dict]],
        lexical_results: List[Tuple[float, Dict]],
        final_k: int = 4,
        fusion_k: int = 60,
    ) -> List[Dict]:
        by_id: Dict[str, Dict] = {}

        def _key(meta: Dict, fallback_rank: int, source: str) -> str:
            return str(meta.get("hash") or meta.get("id") or f"{source}:{fallback_rank}")

        for rank, (_, meta) in enumerate(dense_results, start=1):
            key = _key(meta, rank, "dense")
            if key not in by_id:
                by_id[key] = {"meta": meta, "rrf": 0.0, "dense_rank": None, "lexical_rank": None}
            by_id[key]["rrf"] += 1.0 / (fusion_k + rank)
            by_id[key]["dense_rank"] = rank

        for rank, (_, meta) in enumerate(lexical_results, start=1):
            key = _key(meta, rank, "lex")
            if key not in by_id:
                by_id[key] = {"meta": meta, "rrf": 0.0, "dense_rank": None, "lexical_rank": None}
            by_id[key]["rrf"] += 1.0 / (fusion_k + rank)
            by_id[key]["lexical_rank"] = rank

        ranked = sorted(by_id.values(), key=lambda x: x["rrf"], reverse=True)
        return [
            {
                **x["meta"],  # copy meta, don't mutate original
                "score": float(x["rrf"]),
                "dense_rank": x["dense_rank"],
                "lexical_rank": x["lexical_rank"],
            }
            for x in ranked[:final_k]
        ]

    def retrieve(self, query: str, k: int = 4, dense_k: int = 5, lexical_k: int = 5) -> List[Dict]:
        t0 = time.time()

        qv = self.embedder.embed(query)
        dense_results = self.store.search(qv, k=dense_k)
        logger.info(
            "Dense results: count=%d top=%s",
            len(dense_results),
            [{"title": m.get("title"), "section": m.get("section"), "score": round(float(s), 4)} for s, m in dense_results[:3]],
        )
        lexical_results = self._lexical_search(query, k=lexical_k)
        logger.info(
            "Lexical results: count=%d top=%s",
            len(lexical_results),
            [{"title": m.get("title"), "section": m.get("section"), "score": round(float(s), 4)} for s, m in lexical_results[:3]],
        )

        fused = self._rrf_fuse(
            dense_results=dense_results,
            lexical_results=lexical_results,
            final_k=k,
        )

        logger.info(
            "Fused search results: count=%d top=%s",
            len(fused),
            [{"title": m.get("title"), "section": m.get("section"), "score": round(float(m.get("score", 0.0)), 4), "dense_rank": m.get("dense_rank"), "lexical_rank": m.get("lexical_rank")} for m in fused[:3]],
        )
        self.metrics.add_retrieval((time.time() - t0) * 1000.0)
        return fused

    def generate(self, query: str, contexts: List[Dict]) -> str:
        t0 = time.time()
        try:
            answer = self.llm.generate(query, contexts)
        except Exception as e:
            # Runtime provider failure (401/network/etc): fall back to deterministic stub
            self.llm = StubLLM()
            self.llm_name = "stub"
            answer = self.llm.generate(query, contexts)
            
        self.metrics.add_generation((time.time()-t0)*1000.0)
        return answer

    def stats(self) -> Dict:
        m = self.metrics.summary()
        return {
            "total_docs": len(self._doc_titles),
            "total_chunks": self._chunk_count,
            "embedding_model": settings.embedding_model,
            "llm_model": self.llm_name,
            **m
        }

    def assess_confidence(self, contexts: List[Dict]) -> Dict:
        scores = [float(c.get("score", 0.0)) for c in contexts]
        top_score = scores[0]
        second_score = scores[1] if len(scores) > 1 else 0.0
        score_gap = top_score - second_score

        titles = [c.get("title", "") for c in contexts if c.get("title")]
        source_diversity = len(set(titles))

        # RRF score reference: max is around 0.0328 when rank 1 in both lists
        weak_top = top_score < 0.015
        flat_ranking = len(scores) > 1 and score_gap < 0.0015
        conflicting_sources = source_diversity >= 2 and flat_ranking

        if weak_top:
            level = "low"
            reason = "weak_top_score"
        elif conflicting_sources:
            level = "conflict"
            reason = "flat_scores_across_multiple_sources"
        else:
            level = "high"
            reason = "sufficient_signal"

        confidence =  {
            "level": level,
            "needs_clarification": level in ["low", "conflict"],
            "reason": reason,
            "top_score": round(top_score, 4),
            "score_gap": round(score_gap, 4),
            "source_diversity": source_diversity,
        }

        logger.info(f"Confidence assessment: {confidence} for contexts: {titles} with scores: {scores}")

        return confidence

    def build_clarifying_question(self, contexts: List[Dict]) -> str:
        top_titles = []
        for c in contexts[:3]:
            t = c.get("title")
            if t and t not in top_titles:
                top_titles.append(t)

        if len(top_titles) >= 2:
            return (
                f"I found related info across {', '.join(top_titles[:2])}, etc. "
                "Do you want return-policy guidance, warranty guidance, delivery guidance, product recall guidance, or something else?"
            )

        return "Could you clarify what type of guidance you're looking for, such as return policy, warranty, delivery, product recall, or something else?"

    def build_abstain_answer(self, contexts: List[Dict], confidence: Dict) -> str:
        logger.info(f"Building abstain answer due to confidence level: {confidence['level']} with reason: {confidence['reason']}")
        cq = self.build_clarifying_question(contexts)
        return (
            "I am not fully confident the retrieved policy evidence is strong enough for a definitive answer.\n\n"
            f"Reason: {confidence.get('reason', 'insufficient_signal')}.\n"
            f"{cq}"
        )

# ---- Helpers ----
def build_chunks_from_docs(docs: List[Dict], chunk_size: int, overlap: int) -> List[Dict]:
    out = []
    for d in docs:
        for ch in chunk_text(d["text"], chunk_size, overlap):
            out.append({"title": d["title"], "section": d["section"], "text": ch})
    logger.info(f"Chunks built from docs: total_chunks={len(out)} sample={out[:2]}")
    return out

