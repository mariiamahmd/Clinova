from src.preprocessing import extract_text, split_paragraphs, create_chunks, create_sections
import chromadb
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import json

PDF1 = "data/skin-cancer.pdf"
PDF2 = "data/skin-cancer-prevention.pdf"

pdf1 = extract_text(PDF1)
pdf2 = extract_text(PDF2)

para1 = split_paragraphs(pdf1)
para2 = split_paragraphs(pdf2)

sect1 = create_sections(para1)
sect2 = create_sections(para2)

chunk1 = create_chunks(sect1, "skin_cancer")
chunk2 = create_chunks(sect2, "skin_prevention")

all_chunks = chunk1 + chunk2


documents = [chunk["text"] for chunk in all_chunks]
ids = [chunk["chunk_id"] for chunk in all_chunks]

metadatas = [
    {
        "document": chunk["document"],
        "section": chunk["section"],
        "page": ",".join(map(str, chunk["page"])),
        "chunk_id": chunk["chunk_id"]
    }
    for chunk in all_chunks
]

embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

embeddings = embedding_model.encode(documents, normalize_embeddings=True, show_progress_bar=True).tolist()

tokenized_documents = [document.lower().split() for document in documents]
BM25Okapi(tokenized_documents)

client = chromadb.PersistentClient(path="chroma_db")

try:
    client.delete_collection("skin_cancer")
except Exception:
    pass

collection = client.get_or_create_collection(
    name="skin_cancer",
    configuration={"hnsw": {"space": "cosine"}}
)

collection.add(ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas)

with open("bm25_data.json", "w", encoding="utf-8") as f:
    json.dump({
        "documents": documents,
        "ids": ids,
        "metadatas": metadatas,
        "tokenized_documents": tokenized_documents
    }, f)

print(f"Indexed {len(all_chunks)} chunks.")