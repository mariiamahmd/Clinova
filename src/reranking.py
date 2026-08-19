from sentence_transformers import CrossEncoder

reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

def rerank_results(query, results, top_k=5):
    if not results:
        return []

    pairs = [[query, result["document"]] for result in results]
    scores = reranker.predict(pairs)

    reranked = []

    for result, score in zip(results, scores):
        new_result = result.copy()
        new_result["reranker_score"] = float(score)
        reranked.append(new_result)

    reranked.sort(key=lambda x: x["reranker_score"], reverse=True)

    return reranked[:top_k]