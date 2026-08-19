from src.retrieval import hybrid_search
from src.reranking import rerank_results
from src.generation import build_context, generate_answer_from_context, rewrite_query

def retrieve(query, candidate_k=20, final_k=5):
    candidates = hybrid_search(query=query, top_k=candidate_k, semantic_k=candidate_k, keyword_k=candidate_k)
    return rerank_results(query=query, results=candidates, top_k=final_k)

def determine_confidence(results):
    if not results:
        return "Insufficient"

    top_score = results[0].get("reranker_score")

    if top_score is None:
        return "Insufficient"
    if top_score >= 3.0:
        return "High"
    if top_score >= 1.0:
        return "Medium"

    return "Insufficient"

def build_citations(results):
    return [
        {
            "document": result["metadata"]["document"],
            "section": result["metadata"].get("section", "Unknown"),
            "page": result["metadata"]["page"],
            "chunk_id": result["chunk_id"]
        }
        for result in results
    ]
def generate_answer(query, candidate_k=20, final_k=5, chat_history=None):

    if chat_history is None:
        chat_history = []

    if query.strip().lower() in [
        "who are you",
        "who are you?",
        "what are you",
        "what are you?"
    ]:
        return (
            "I am a helpful clinical evidence assistant. Ask me what you want.",
            [],
            "High",
            []
        )

    # 1. Rewrite follow-up question using chat history
    search_query = rewrite_query(
        query,
        chat_history
    )

    print("\n==============================")
    print("Original query:", query)
    print("Rewritten query:", search_query)
    print("==============================")

    # 2. Retrieve using rewritten query
    results = retrieve(
        search_query,
        candidate_k,
        final_k
    )

    # 3. Debug results
    for i, result in enumerate(results, start=1):
        print(
            f"Result {i}:",
            result.get("reranker_score"),
            result.get("chunk_id")
        )

    # 4. Determine confidence
    confidence = determine_confidence(results)

    # IMPORTANT:
    # Do NOT stop here if confidence is insufficient.
    # Let the LLM inspect the retrieved evidence.

    # 5. Build evidence context
    context = build_context(results)

    # 6. Generate grounded answer
    answer = generate_answer_from_context(
        query,
        context,
        chat_history
    )

    # 7. Let the answer determine whether evidence was sufficient
    if answer.strip() == (
        "The retrieved evidence is insufficient to answer this question."
    ):
        return (
            answer,
            [],
            "Insufficient",
            []
        )

    # 8. If LLM answered, return sources
    return (
        answer,
        results,
        confidence,
        build_citations(results)
    )