import json
import os
import shutil
from pathlib import Path

from bs4 import BeautifulSoup
from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# Paths
CHROMA_PATH = "data/chroma_db"
DATA_FILE_JSONL = Path("data/raw_docs.jsonl")
DATA_FILE_JSON = Path("data/raw_docs.json")
CHUNKS_FILE = Path("data/chunks.jsonl")  # <-- Hybrid (BM25) ამისთვის

# Chunking
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Skip tiny docs
MIN_CONTENT_LEN = 60


def clean_html(html_text: str) -> str:
    if not html_text:
        return ""
    soup = BeautifulSoup(html_text, "lxml")
    return soup.get_text(separator="\n", strip=True)


def load_raw_docs() -> list:
    """
    კითხულობს JSONL-ს თუ არსებობს, თუ არა — JSON-ს.
    """
    if DATA_FILE_JSONL.exists():
        docs = []
        with DATA_FILE_JSONL.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    docs.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return docs

    if DATA_FILE_JSON.exists():
        try:
            return json.loads(DATA_FILE_JSON.read_text(encoding="utf-8"))
        except Exception:
            return []

    return []


def export_chunks_jsonl(chunks: list[Document], out_path: Path):
    """
    BM25Retriever-სთვის ვიწერთ chunk-ებს ფაილად (JSONL).
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for d in chunks:
            f.write(
                json.dumps(
                    {"page_content": d.page_content, "metadata": d.metadata},
                    ensure_ascii=False,
                )
                + "\n"
            )
    print(f"📝 chunks შენახულია: {out_path}")


def main():
    if not os.getenv("GOOGLE_API_KEY"):
        print("❌ GOOGLE_API_KEY ვერ მოიძებნა .env-ში!")
        print("👉 დაამატე .env ფაილში: GOOGLE_API_KEY=...")
        return

    raw_docs = load_raw_docs()
    if not raw_docs:
        print("❌ დასამუშავებელი დოკუმენტები არ მოიძებნა.")
        print("👉 ჯერ გაუშვი: python src/download_data.py")
        return

    print(f"📄 ჩაიტვირთა {len(raw_docs)} დოკუმენტი")

    langchain_docs: list[Document] = []
    skipped_short = 0

    for item in raw_docs:
        if not isinstance(item, dict):
            continue

        title = (item.get("title") or item.get("name") or "").strip()
        url = (item.get("url") or "").strip()
        doc_id = item.get("id")
        uuid = item.get("uuid")

        raw_html = (
            item.get("text_html")
            or item.get("additionalDescription")
            or item.get("description")
            or item.get("text")
            or ""
        )

        text = clean_html(raw_html)
        content = f"{title}\n\n{text}".strip()

        if len(content) < MIN_CONTENT_LEN:
            skipped_short += 1
            continue

        langchain_docs.append(
            Document(
                page_content=content,
                metadata={
                    "source": url,
                    "title": title,
                    "doc_id": str(doc_id) if doc_id is not None else "",
                    "uuid": str(uuid) if uuid is not None else "",
                    "createDate": str(item.get("createDate") or ""),
                    "updateDate": str(item.get("updateDate") or ""),
                    "species": str(item.get("species") or ""),
                },
            )
        )

    if not langchain_docs:
        print("❌ ვერცერთი ვალიდური დოკუმენტი ვერ მოიძებნა.")
        return

    print(f"✅ ვალიდური დოკუმენტები: {len(langchain_docs)} | გამოტოვებული მოკლე: {skipped_short}")

    print("✂️ ტექსტის დაჭრა chunks-ებად...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(langchain_docs)
    print(f"📦 chunks რაოდენობა: {len(chunks)}")

    # Hybrid-ისთვის chunks ფაილში შენახვა
    export_chunks_jsonl(chunks, CHUNKS_FILE)

    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

    # Rebuild Chroma
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
        print("🧹 ძველი chroma_db წაიშალა")

    print("💾 ჩაწერა ChromaDB-ში...")
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH,
    )

    print("🎉 დასრულდა! ბაზა შეიქმნა:", CHROMA_PATH)


if __name__ == "__main__":
    main()