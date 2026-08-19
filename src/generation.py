import os
from openai import OpenAI

from dotenv import load_dotenv

load_dotenv()

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

client = OpenAI(
    api_key=MISTRAL_API_KEY,
    base_url="https://api.mistral.ai/v1"
)

MODEL = "mistral-small-latest"

def rewrite_query(query, chat_history=None):

    if chat_history is None:
        chat_history = []

    history_text = ""

    for message in chat_history:
        history_text += (
            f'{message["role"].capitalize()}: '
            f'{message["content"]}\n'
        )

    messages = [
        {
            "role": "system",
            "content": """
You rewrite follow-up questions into standalone search queries.

Use the conversation history to understand what the user is referring to.

If the current question contains words such as:
- it
- this
- that
- they
- them
- its
- their
- these
- those
- benefits
- risks
- effects
- treatment
- prevention

resolve the reference using the previous conversation.

The rewritten query MUST preserve the meaning of the user's
current question.

Do NOT answer the question.

Return ONLY the standalone search query.
"""
        },
        {
            "role": "user",
            "content": f"""
Conversation history:

{history_text}

Current question:

{query}

Standalone search query:
"""
        }
    ]

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0,
        max_tokens=100
    )

    return response.choices[0].message.content.strip()

# takes retreived data to sent to the llm
def build_context(results):
    context = []

    for result in results:
        metadata = result["metadata"]

        context.append(
            f"""Document: {metadata.get("document", "Unknown")}
Section: {metadata.get("section", "Unknown")}
Page: {metadata.get("page", "Unknown")}

Evidence:
{result["document"]}"""
        )

    return "\n\n".join(context)


# LLM roledef 
def generate_answer_from_context(query, context, chat_history=None):

    if chat_history is None:
        chat_history = []

    history_text = ""

    for message in chat_history:
        history_text += (
            f'{message["role"].capitalize()}: '
            f'{message["content"]}\n'
        )

    messages = [
        {
            "role": "system",
            "content": """You are a helpful clinical evidence assistant.

Answer the user's question naturally and directly.

Use ONLY the information contained in the provided evidence as factual support.

You may use the conversation history to understand what the user is referring to,
such as pronouns, abbreviations, or follow-up questions.

Do NOT use outside medical knowledge.
Do NOT invent facts, recommendations, or sources.

If the evidence does NOT contain enough information to answer the question, reply exactly:

The retrieved evidence is insufficient to answer this question.

Do not talk about retrieval, chunks, embeddings, reranking,
confidence scores, or the system."""
        },
        {
            "role": "user",
            "content": f"""Conversation history:

{history_text}

Current question:

{query}

Retrieved evidence:

{context}

Answer:"""
        }
    ]

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0,
        max_tokens=200
    )

    return response.choices[0].message.content.strip()