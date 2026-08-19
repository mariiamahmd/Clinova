import re
from pypdf import PdfReader
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")

def clean_text(text):
    if not text:
        return ""

    text = text.replace("\xa0", " ").replace("\r\n", "\n").replace("\r", "\n")

    text = re.sub(r"©\s*NICE\s+\d{4}\.\s*All\s+rights\s+reserved\.\s*Subject\s+to\s+Notice\s+of\s+rights\s*\(\s*https://www\.nice\.org\.uk/terms-and-?(?:\s*\n\s*)?conditions#notice-of-rights\s*\)\.?", "", text, flags=re.IGNORECASE)
    text = re.sub(r"©\s*\d{4}\s+American\s+Medical\s+Association\.\s*All\s+rights\s+reserved\.", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\d+\s+JAMA\s+[A-Za-z]+\s+\d{1,2},\s+\d{4}\s+Volume\s+\d+,\s+Number\s+\d+\s+\(Reprinted\)\s+jama\.com", "", text, flags=re.IGNORECASE)
    text = re.sub(r"jama\.com\s+\(Reprinted\)\s+JAMA\s+[A-Za-z]+\s+\d{1,2},\s+\d{4}\s+Volume\s+\d+,\s+Number\s+\d+\s+\d+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"Page\s+\d+\s+of\s*\n?\s*\d+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^\s*\d{1,4}\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*Skin\s+cancer\s*$", "", text, flags=re.MULTILINE | re.IGNORECASE)
    text = re.sub(r"^\s*Skin\s+cancer\s+prevention\s+\(PH32\)\s*$", "", text, flags=re.MULTILINE | re.IGNORECASE)
    text = re.sub(r"-\s*\n\s*", "", text)

    text = text.replace("－", "-").replace("•", "-")
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()

def extract_text(pdf_path):
    reader = PdfReader(pdf_path)
    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        if "skin_prevention" in pdf_path.lower() and page_number >= 21:
            break

        text = page.extract_text() or ""
        cleaned = clean_text(text)

        if cleaned:
            pages.append({"page": page_number, "text": cleaned})

    return pages

def split_paragraphs(pages):
    paragraphs = []

    for page in pages:
        for paragraph in page["text"].split("\n\n"):
            paragraph = paragraph.strip()

            if paragraph:
                paragraphs.append({"page": page["page"], "text": paragraph})

    return paragraphs

def create_sections(paragraphs, paragraphs_per_section=5):
    return [paragraphs[i:i + paragraphs_per_section] for i in range(0, len(paragraphs), paragraphs_per_section)]

def create_chunks(sections, document_name, chunk_size=300, overlap=0.15):
    chunks = []
    overlap_tokens = int(chunk_size * overlap)
    step = chunk_size - overlap_tokens
    chunk_id = 1

    for section_id, section in enumerate(sections, start=1):
        section_text = " ".join(paragraph["text"] for paragraph in section)
        pages = list(dict.fromkeys(paragraph["page"] for paragraph in section))

        tokens = tokenizer(section_text, add_special_tokens=False, truncation=False)["input_ids"]

        for start in range(0, len(tokens), step):
            chunk_tokens = tokens[start:start + chunk_size]

            if not chunk_tokens:
                break

            chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)

            chunks.append({
                "document": document_name,
                "section": section_id,
                "page": pages,
                "chunk_id": f"{document_name}_s{section_id}_c{chunk_id}",
                "text": chunk_text
            })

            chunk_id += 1

    return chunks