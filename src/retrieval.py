import json
import re
import chromadb
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_collection("skin_cancer")

embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

with open("bm25_data.json", "r", encoding="utf-8") as f:
    bm25_data = json.load(f)

documents = bm25_data["documents"]
ids = bm25_data["ids"]

def tokenize(text):
    return re.findall(r"\b[a-zA-Z0-9]+\b", text.lower())

tokenized_documents = [tokenize(document) for document in documents]
bm25 = BM25Okapi(tokenized_documents)

def hybrid_search(query, top_k=20, semantic_k=20, keyword_k=20):
    query_tokens = tokenize(query)
    bm25_scores = bm25.get_scores(query_tokens)
    keyword_indices = bm25_scores.argsort()[::-1][:keyword_k]

    query_embedding = embedding_model.encode([query], normalize_embeddings=True).tolist()

    semantic_results = collection.query(query_embeddings=query_embedding, n_results=semantic_k, include=["documents", "metadatas", "distances"])

    semantic_ids = semantic_results["ids"][0]
    semantic_distances = semantic_results["distances"][0]

    semantic_similarity = {chunk_id: 1 - distance for chunk_id, distance in zip(semantic_ids, semantic_distances)}

    rrf_k = 60
    rrf_scores = {}

    for rank, idx in enumerate(keyword_indices):
        chunk_id = ids[idx]
        rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + 1 / (rrf_k + rank + 1)

    for rank, chunk_id in enumerate(semantic_ids):
        rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + 1 / (rrf_k + rank + 1)

    ranked_chunks = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    final_ids = [chunk_id for chunk_id, _ in ranked_chunks[:top_k]]

    final_results = collection.get(ids=final_ids, include=["documents", "metadatas"])

    result_map = {
        chunk_id: {"document": document, "metadata": metadata}
        for chunk_id, document, metadata in zip(final_results["ids"], final_results["documents"], final_results["metadatas"])
    }

    return [
        {
            "chunk_id": chunk_id,
            "semantic_similarity": semantic_similarity.get(chunk_id),
            "rrf_score": rrf_score,
            "document": result_map[chunk_id]["document"],
            "metadata": result_map[chunk_id]["metadata"]
        }
        for chunk_id, rrf_score in ranked_chunks[:top_k]
    ]