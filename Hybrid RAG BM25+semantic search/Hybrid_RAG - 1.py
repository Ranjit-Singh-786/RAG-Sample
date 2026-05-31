from dataclasses import dataclass
from typing import List, Dict, Tuple
import numpy as np
import faiss

from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


# =========================================================
# CONFIG
# =========================================================

EMBED_MODEL = "all-MiniLM-L6-v2"
TOP_K = 3

# Hybrid weights
BM25_WEIGHT = 0.4
SEMANTIC_WEIGHT = 0.6


# =========================================================
# DATA MODEL
# =========================================================

@dataclass
class Document:
    doc_id: int
    text: str
    metadata: Dict


# =========================================================
# SAMPLE DOCUMENTS
# =========================================================

documents = [
    Document(
        doc_id=1,
        text="Python is a popular programming language used for AI and backend development.",
        metadata={"category": "programming"},
    ),
    Document(
        doc_id=2,
        text="LangGraph helps developers build agentic AI workflows with state management.",
        metadata={"category": "ai"},
    ),
    Document(
        doc_id=3,
        text="FAISS is a high-performance vector database library developed by Meta.",
        metadata={"category": "vector-db"},
    ),
    Document(
        doc_id=4,
        text="BM25 is a keyword-based ranking algorithm used in search engines.",
        metadata={"category": "search"},
    ),
    Document(
        doc_id=5,
        text="Hybrid RAG combines semantic retrieval and keyword retrieval for better accuracy.",
        metadata={"category": "rag"},
    ),
]


# =========================================================
# HYBRID RAG ENGINE
# =========================================================

class HybridRAG:

    def __init__(self, docs: List[Document]):

        self.docs = docs

        # ---------- TEXT LIST ----------
        self.texts = [doc.text for doc in docs]

        # ---------- BM25 ----------
        tokenized_docs = [text.lower().split() for text in self.texts]
        self.bm25 = BM25Okapi(tokenized_docs)

        # ---------- EMBEDDING MODEL ----------
        self.embedding_model = SentenceTransformer(EMBED_MODEL)

        # ---------- VECTOR EMBEDDINGS ----------
        self.doc_embeddings = self.embedding_model.encode(
            self.texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        # ---------- FAISS INDEX ----------
        embedding_dim = self.doc_embeddings.shape[1]

        self.index = faiss.IndexFlatIP(embedding_dim)
        self.index.add(self.doc_embeddings)

    # =====================================================
    # NORMALIZATION
    # =====================================================

    def min_max_normalize(self, scores):

        scores = np.array(scores, dtype=np.float32)

        min_score = scores.min()
        max_score = scores.max()

        # avoid divide-by-zero
        if max_score - min_score == 0:
            return np.ones_like(scores)

        normalized = (
            (scores - min_score)
            / (max_score - min_score)
        )

        return normalized

    # =====================================================
    # BM25 SEARCH
    # =====================================================

    def keyword_search(
        self,
        query: str,
        top_k: int = TOP_K,
    ) -> List[Tuple[int, float]]:

        tokens = query.lower().split()

        scores = self.bm25.get_scores(tokens)

        # normalize scores
        normalized_scores = self.min_max_normalize(scores)

        ranked = np.argsort(normalized_scores)[::-1][:top_k]

        return [
            (idx, float(normalized_scores[idx]))
            for idx in ranked
        ]

    # =====================================================
    # SEMANTIC SEARCH
    # =====================================================

    def semantic_search(
        self,
        query: str,
        top_k: int = TOP_K,
    ) -> List[Tuple[int, float]]:

        query_embedding = self.embedding_model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        scores, indices = self.index.search(
            query_embedding,
            len(self.docs),
        )

        semantic_scores = scores[0]

        # normalize semantic scores

        

        ranked_indices = indices[0][:top_k]
        ranked_scores = semantic_scores[:top_k]

        results = []

        for idx, score in zip(
            ranked_indices,
            ranked_scores,
        ):
            results.append((int(idx), float(score)))

        return results

    # =====================================================
    # HYBRID SEARCH
    # =====================================================

    def hybrid_search(
        self,
        query: str,
        top_k: int = TOP_K,
    ) -> List[Dict]:

        keyword_results = self.keyword_search(
            query,
            top_k=len(self.docs),
        )

        semantic_results = self.semantic_search(
            query,
            top_k=len(self.docs),
        )

        combined_scores = {}

        # ---------- BM25 ----------
        for idx, score in keyword_results:

            combined_scores[idx] = (
                combined_scores.get(idx, 0)
                + score * BM25_WEIGHT
            )

        # ---------- Semantic ----------
        for idx, score in semantic_results:

            combined_scores[idx] = (
                combined_scores.get(idx, 0)
                + score * SEMANTIC_WEIGHT
            )

        # ---------- Final Ranking ----------
        ranked = sorted(
            combined_scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:top_k]

        results = []

        for idx, final_score in ranked:

            doc = self.docs[idx]

            results.append({
                "doc_id": doc.doc_id,
                "score": round(final_score, 4),
                "text": doc.text,
                "metadata": doc.metadata,
            })

        return results


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    rag = HybridRAG(documents)

    query = "How does keyword retrieval work in search systems?"

    print("\n" + "=" * 60)
    print("QUERY:")
    print(query)
    print("=" * 60)

    # -----------------------------------------------------
    # HYBRID SEARCH
    # -----------------------------------------------------

    print("\n[HYBRID SEARCH RESULTS]\n")

    hybrid_results = rag.hybrid_search(query)

    for item in hybrid_results:

        print(f"Final Score: {item['score']}")
        print(f"Doc ID: {item['doc_id']}")
        print(f"Text: {item['text']}")
        print(f"Metadata: {item['metadata']}")
        print("-" * 50)