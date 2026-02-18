import os
import re
import json
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

from langchain_community.retrievers import BM25Retriever

load_dotenv()

CHROMA_PATH = "data/chroma_db"
CHUNKS_FILE = Path("data/chunks.jsonl")

st.set_page_config(page_title="InfoHub AI", page_icon="🇬🇪")

# -----------------------------
# Defaults
# -----------------------------
DEFAULT_K = 5
DEFAULT_THRESHOLD = 0.55

SOURCE_FOOTER = (
    "წყარო: საინფორმაციო და მეთოდოლოგიური ჰაბზე განთავსებული დოკუმენტების მიხედვით "
    "(საგადასახადო და საბაჟო ადმინისტრირებასთან დაკავშირებული დოკუმენტები და ინფორმაცია ერთ სივრცეში)"
)

# -----------------------------
# Smalltalk filter
# -----------------------------
GREETING_PATTERNS = [
    r"^\s*გამარჯობა\s*!?\s*$",
    r"^\s*სალამი\s*!?\s*$",
    r"^\s*ჰეი\s*!?\s*$",
    r"^\s*დილამშვიდობისა\s*!?\s*$",
    r"^\s*საღამომშვიდობისა\s*!?\s*$",
    r"^\s*მადლობა\s*!?\s*$",
    r"^\s*ok\s*!?\s*$",
    r"^\s*okay\s*!?\s*$",
    r"^\s*hello\s*!?\s*$",
    r"^\s*hi\s*!?\s*$",
]

def is_smalltalk(query: str) -> bool:
    q = (query or "").strip().lower()
    for p in GREETING_PATTERNS:
        if re.match(p, q, flags=re.IGNORECASE):
            return True
    if len(q) <= 6:
        return True
    if re.fullmatch(r"[\W_]+", q or ""):
        return True
    return False

def format_docs(docs):
    return "\n\n---\n\n".join(d.page_content for d in docs)

def load_chunks_from_file(path: Path) -> list[Document]:
    docs = []
    if not path.exists():
        return docs
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            docs.append(
                Document(
                    page_content=obj.get("page_content", "") or "",
                    metadata=obj.get("metadata", {}) or {},
                )
            )
    return docs

def hybrid_retrieve(query: str, bm25_retriever, vectorstore: Chroma, k: int):
    """
    Manual Hybrid:
    - BM25 top-k
    - Vector top-k
    - merge + dedupe
    """
    bm25_docs = bm25_retriever.invoke(query) if bm25_retriever else []
    bm25_docs = bm25_docs[:k]

    vector_docs = vectorstore.similarity_search(query, k=k)

    combined = []
    seen = set()

    # BM25 first (keyword priority), then vector
    for d in bm25_docs + vector_docs:
        src = d.metadata.get("source", "") or ""
        title = d.metadata.get("title", "") or ""
        key = (src + "|" + title + "|" + d.page_content[:120]).strip()
        if key in seen:
            continue
        seen.add(key)
        combined.append(d)

    return combined[:k]

@st.cache_resource
def load_chain(k: int):
    if not os.path.exists(CHROMA_PATH):
        return None, None, None, False, 0

    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
    vectorstore = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)

    # BM25 from chunks.jsonl
    bm25_docs = load_chunks_from_file(CHUNKS_FILE)
    bm25_retriever = None
    if bm25_docs:
        bm25_retriever = BM25Retriever.from_documents(bm25_docs)
        bm25_retriever.k = k

    using_hybrid = bm25_retriever is not None

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0)

    template = f"""
You are an assistant for the Revenue Service of Georgia.
Answer the question based ONLY on the context below.

Rules:
1. Answer in Georgian.
2. If the answer is not in the context, say you don't know.
3. ALWAYS end the answer with this exact phrase:
   "{SOURCE_FOOTER}"

Context:
{{context}}

Question:
{{question}}
"""
    prompt = ChatPromptTemplate.from_template(template)

    chain = (
        {
            "context": RunnableLambda(lambda q: format_docs(hybrid_retrieve(q, bm25_retriever, vectorstore, k))),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain, bm25_retriever, vectorstore, using_hybrid, len(bm25_docs)

# -----------------------------
# UI
# -----------------------------
st.title("InfoHub AI Agent 🏛️")

st.sidebar.header("⚙️ პარამეტრები / Debug")
debug_mode = st.sidebar.checkbox("Debug რეჟიმი", value=True)
k = st.sidebar.slider("Top-K დოკუმენტი (retrieval)", min_value=1, max_value=10, value=DEFAULT_K)
threshold = st.sidebar.slider("Threshold (წყაროების ჩვენება)", min_value=0.10, max_value=0.95, value=DEFAULT_THRESHOLD, step=0.01)

chain, bm25_retriever, vectorstore, using_hybrid, bm25_count = load_chain(k)

if not chain:
    st.error("Database not found. Please run `python src/ingest.py` first.")
    st.stop()

st.sidebar.caption(f"Retriever: {'HYBRID (BM25 + Vector)' if using_hybrid else 'Vector-only (BM25 chunks not found)'}")
if using_hybrid:
    st.sidebar.caption(f"BM25 chunks loaded: {bm25_count}")
else:
    st.sidebar.caption("BM25 chunks not found → run `python src/ingest.py` to create data/chunks.jsonl")

# Chat state
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if query := st.chat_input("დასვით კითხვა..."):
    st.chat_message("user").write(query)
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("assistant"):
        # Smalltalk: no retrieval, no sources, no debug
        if is_smalltalk(query):
            response = (
                "გამარჯობა! 😊\n\n"
                "დასვით კითხვა საგადასახადო ან საბაჟო ადმინისტრირების თემაზე და გიპასუხებთ InfoHub-ის დოკუმენტებზე დაყრდნობით.\n\n"
                f"{SOURCE_FOOTER}"
            )
            st.write(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.stop()

        # Hybrid docs for sources + debug view
        hybrid_docs = hybrid_retrieve(query, bm25_retriever, vectorstore, k)

        # Vector scores only for deciding whether sources should be meaningful
        scored = vectorstore.similarity_search_with_relevance_scores(query, k=k)
        best_score = max([score for _, score in scored], default=0)

        # Answer
        response = chain.invoke(query)
        st.write(response)

        # -------- Sources (ALWAYS visible as UI section, independent of debug) --------
        # BUT: links are shown only if best_score >= threshold (as you wanted originally)
        with st.expander("წყაროს ლინკები", expanded=True):
            if best_score >= threshold:
                sources = []
                for d in hybrid_docs:
                    src = d.metadata.get("source")
                    if src and src not in sources:
                        sources.append(src)

                if sources:
                    for s in sources:
                        st.write(s)
                else:
                    st.write("ამ შეკითხვაზე შესაბამისი წყაროები ვერ მოიძებნა (Top-K შედეგებში).")
            else:
                st.write("შესაბამისობა (score) დაბალია, ამიტომ წყაროები არ გამოჩნდა ამ პასუხისთვის.")

        # -------- Debug (ONLY when debug_mode ON) --------
        if debug_mode:
            with st.expander("🧪 Debug: retrieval შედეგები", expanded=False):
                st.write(f"**Vector best_score:** `{best_score:.3f}`")
                st.write(f"**Top-K:** `{k}`")
                st.write(f"**Threshold:** `{threshold:.2f}`")
                st.write(f"**Retriever:** {'HYBRID' if using_hybrid else 'Vector-only'}")

                st.markdown("### Hybrid – Top Documents")
                for idx, d in enumerate(hybrid_docs, start=1):
                    title = d.metadata.get("title", "")
                    doc_id = d.metadata.get("doc_id", "")
                    uuid = d.metadata.get("uuid", "")
                    source = d.metadata.get("source", "")

                    st.markdown(f"**{idx})** {title if title else '(no title)'}")
                    if doc_id:
                        st.write(f"doc_id: {doc_id}")
                    if uuid:
                        st.write(f"uuid: {uuid}")
                    if source:
                        st.write(f"source: {source}")

                    preview = (d.page_content or "").strip()
                    if len(preview) > 500:
                        preview = preview[:500] + " …"
                    st.code(preview)
                    st.divider()

        st.session_state.messages.append({"role": "assistant", "content": response})