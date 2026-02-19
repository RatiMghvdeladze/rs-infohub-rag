"""
InfoHub AI Agent — Streamlit RAG Application.

Major improvements:
  - Uses the unified retriever module (RRF + multi-query)
  - Conversation-aware prompting (last N messages as context)
  - Upgraded LLM (gemini-2.5-flash)
  - Better prompt with chain-of-thought reasoning
  - Single retrieval call per question (no double-fetch)
  - Improved source display with titles
"""

import os
import re

import streamlit as st
from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from pathlib import Path
from langchain_community.retrievers import BM25Retriever

from retriever import load_chunks_from_file, retrieve


load_dotenv()

CHROMA_PATH = "data/chroma_db"
CHUNKS_FILE = "data/chunks.jsonl"

st.set_page_config(page_title="InfoHub AI", page_icon="🇬🇪")

# -----------------------------
# Defaults
# -----------------------------
DEFAULT_K = 10
DEFAULT_THRESHOLD = 0.25
MAX_HISTORY_TURNS = 4  # how many past Q&A pairs to include in prompt

SOURCE_FOOTER = (
    "წყარო: საინფორმაციო და მეთოდოლოგიური ჰაბზე განთავსებული დოკუმენტების მიხედვით "
    "(საგადასახადო და საბაჟო ადმინისტრირებასთან დაკავშირებული დოკუმენტები და ინფორმაცია ერთ სივრცეში)"
)

# -----------------------------
# Smalltalk filter
# -----------------------------
GREETING_PATTERNS = [
    r"^\s*გამარჯობა[!.\s]*$",
    r"^\s*გაუმარჯოს[!.\s]*$",
    r"^\s*სალამი[!.\s]*$",
    r"^\s*ჰეი[!.\s]*$",
    r"^\s*დილამშვიდობისა[!.\s]*$",
    r"^\s*საღამომშვიდობისა[!.\s]*$",
    r"^\s*მადლობა[!.\s]*$",
    r"^\s*დიდი\s*მადლობა[!.\s]*$",
    r"^\s*გმადლობ(თ|ი)?[!.\s]*$",
    r"^\s*კარგი[!.\s]*$",
    r"^\s*ok[!.\s]*$",
    r"^\s*okay[!.\s]*$",
    r"^\s*hello[!.\s]*$",
    r"^\s*hi[!.\s]*$",
    r"^\s*hey[!.\s]*$",
    r"^\s*thanks[!.\s]*$",
    r"^\s*thank\s*you[!.\s]*$",
    r"^\s*როგორ\s*ხარ[!?.\s]*$",
    r"^\s*რა\s*ხდება[!?.\s]*$",
]

SMALLTALK_RESPONSES = {
    "greeting": (
        "გამარჯობა! 😊\n\n"
        "დასვით კითხვა საგადასახადო ან საბაჟო ადმინისტრირების თემაზე "
        "და გიპასუხებთ InfoHub-ის დოკუმენტებზე დაყრდნობით."
    ),
    "thanks": (
        "არაფრის! 😊 თუ სხვა კითხვა გაქვთ, სიამოვნებით გიპასუხებთ."
    ),
}


def classify_smalltalk(query: str) -> str | None:
    """Return smalltalk category or None if it's a real question."""
    q = (query or "").strip().lower()

    # Very short input
    if len(q) <= 4:
        return "greeting"

    # Only punctuation/symbols
    if re.fullmatch(r"[\W_]+", q):
        return "greeting"

    # Check patterns
    for p in GREETING_PATTERNS:
        if re.match(p, q, flags=re.IGNORECASE):
            # Distinguish thanks vs greeting
            if any(w in q for w in ["მადლობ", "გმადლობ", "thanks", "thank"]):
                return "thanks"
            return "greeting"

    return None


def format_docs(docs):
    return "\n\n---\n\n".join(d.page_content for d in docs)


def build_conversation_context(messages: list, max_turns: int = MAX_HISTORY_TURNS) -> str:
    """Extract the last N Q&A turns from session messages for context."""
    if not messages:
        return ""

    turns = []
    i = len(messages) - 1
    while i >= 0 and len(turns) < max_turns:
        if i >= 1 and messages[i]["role"] == "assistant" and messages[i - 1]["role"] == "user":
            turns.append(
                f"მომხმარებელი: {messages[i-1]['content'][:300]}\n"
                f"ასისტენტი: {messages[i]['content'][:500]}"
            )
            i -= 2
        else:
            i -= 1

    if not turns:
        return ""

    turns.reverse()
    return "წინა დიალოგი:\n" + "\n\n".join(turns)


# ---------------------------------------------------------------------------
# Load resources (cached)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_resources():
    """Load vectorstore, BM25 retriever, and LLM once."""
    if not os.path.exists(CHROMA_PATH):
        return None, None, None, False, 0

    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
    vectorstore = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)

    # Try loading pickled BM25 first (much faster)
    from retriever import load_bm25_retriever
    bm25_retriever = load_bm25_retriever(Path("data/bm25_retriever.pkl"))
    
    # Fallback to building from chunks.jsonl if pickle fails
    bm25_count = 0
    if bm25_retriever is None:
        bm25_docs = load_chunks_from_file(Path(CHUNKS_FILE))
        if bm25_docs:
            bm25_retriever = BM25Retriever.from_documents(bm25_docs)
            bm25_retriever.k = 10
            bm25_count = len(bm25_docs)
    else:
        # We don't know the count easily from pickle without accessing internal list, 
         # but we can just say "Loaded from Pickle"
        bm25_count = -1 

    using_hybrid = bm25_retriever is not None

    # gemini-2.0-flash: fast, no thinking delays, ideal for RAG
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)

    return vectorstore, bm25_retriever, llm, using_hybrid, bm25_count


def generate_answer(question: str, context: str, conversation_history: str, llm) -> str:
    """Generate an answer using the LLM with full context."""
    prompt = f"""შენ ხარ საქართველოს შემოსავლების სამსახურის ინტელექტუალური ასისტენტი.
შენი მიზანია მომხმარებელს დაეხმარო მაქსიმალურად. პასუხი გასცე ქვემოთ მოცემული კონტექსტის საფუძველზე.

წესები:
1. უპასუხე ქართულ ენაზე.
2. ყურადღებით წაიკითხე მთლიანი კონტექსტი. თუ კონტექსტში არის რაიმე დაკავშირებული ინფორმაცია — თუნდაც ნაწილობრივი ან არაპირდაპირი — გამოიყენე და აუხსენი მომხმარებელს. მხოლოდ იმ შემთხვევაში თქვი "ვერ მოიძებნა" თუ კონტექსტში ნამდვილად არაფერია კითხვასთან დაკავშირებული.
3. იყავი ზუსტი და კონკრეტული — მიუთითე კანონის მუხლები, ვადები, განაკვეთები თუ კონტექსტში მოცემულია.
4. პასუხი დააფორმატე გასაგებად: გამოიყენე ჩამონათვალი ან ნუმერაცია საჭიროებისამებრ.
5. თუ კონტექსტში პასუხი ნაწილობრივ არის, მაინც უპასუხე ნაწილობრივად და აღნიშნე რომ სრული ინფორმაციისთვის შემოსავლების სამსახურთან დაკავშირება სჯობს.
6. ყოველთვის დაასრულე პასუხი ზუსტად ამ ფრაზით:
   "{SOURCE_FOOTER}"

{conversation_history}

კონტექსტი:
{context}

კითხვა: {question}"""

    response = llm.invoke(prompt)
    return response.content if hasattr(response, "content") else str(response)


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.title("InfoHub AI Agent 🏛️")

st.sidebar.header("⚙️ პარამეტრები")
debug_mode = st.sidebar.checkbox("Debug რეჟიმი", value=False)
k = st.sidebar.slider("Top-K დოკუმენტი (retrieval)", min_value=1, max_value=20, value=DEFAULT_K) # Increased max K
threshold = st.sidebar.slider(
    "Threshold (წყაროების ჩვენება)",
    min_value=0.10, max_value=0.95, value=DEFAULT_THRESHOLD, step=0.01,
)

vectorstore, bm25_retriever, llm, using_hybrid, bm25_count = load_resources()

if not vectorstore:
    st.error("Database not found. Please run `python src/ingest.py` first.")
    st.stop()

retriever_label = "HYBRID (BM25 + Vector + RRF)" if using_hybrid else "Vector-only"
st.sidebar.caption(f"Retriever: {retriever_label}")
if using_hybrid:
    if bm25_count == -1:
        st.sidebar.caption("BM25 Index: Loaded from Pickle ⚡")
    else:
        st.sidebar.caption(f"BM25 chunks loaded: {bm25_count:,}")
else:
    st.sidebar.caption("BM25 chunks not found → run `python src/ingest.py`")

# Chat state
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if query := st.chat_input("დასვით კითხვა..."):
    st.chat_message("user").write(query)
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("assistant"):
        # Smalltalk check
        smalltalk_type = classify_smalltalk(query)
        if smalltalk_type:
            response = SMALLTALK_RESPONSES.get(smalltalk_type, SMALLTALK_RESPONSES["greeting"])
            st.write(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.stop()

        # === Single unified retrieval call (with Query Expansion) ===
        with st.spinner("🔍 ძიება მიმდინარეობს..."):
            result = retrieve(
                question=query,
                vectorstore=vectorstore,
                bm25_retriever=bm25_retriever,
                llm=llm, # Pass LLM for Query Expansion
                k=k,
            )

        # Build conversation context for follow-up awareness
        conv_history = build_conversation_context(st.session_state.messages[:-1])

        # Generate answer
        context_text = format_docs(result.docs)
        
        if debug_mode:
            st.info(f"Retrieved {len(result.docs)} unique documents.")

        # Generate answer (reliable invoke, no freezing)
        with st.spinner("💭 პასუხის გენერირება..."):
            full_response = generate_answer(query, context_text, conv_history, llm)

        st.markdown(full_response)

        # ------------------------------------------------------------------
        # Sources section
        # ------------------------------------------------------------------
        with st.expander("📄 წყაროები", expanded=False): # Collapsed by default to reduce clutter
            if result.best_vector_score >= threshold and result.docs:
                sources_shown = []
                for d in result.docs:
                    src = d.metadata.get("source", "")
                    title = d.metadata.get("title", "")
                    if src and src not in [s[0] for s in sources_shown]:
                        sources_shown.append((src, title))

                if sources_shown:
                    for src_url, src_title in sources_shown:
                        if src_title:
                            st.markdown(f"• **{src_title}**  \n  {src_url}")
                        else:
                            st.write(f"• {src_url}")
                else:
                    st.write("ამ შეკითხვაზე შესაბამისი წყაროები ვერ მოიძებნა.")
            else:
                st.write("შესაბამისობა (score) დაბალია, ამიტომ წყაროები არ გამოჩნდა ამ პასუხისთვის.")

        # ------------------------------------------------------------------
        # Debug section
        # ------------------------------------------------------------------
        if debug_mode:
            with st.expander("🧪 Debug: Retrieval Details", expanded=False):
                st.write(f"**Best Vector Score:** `{result.best_vector_score:.3f}`")
                st.write(f"**Top-K:** `{k}`")
                st.write(f"**Threshold:** `{threshold:.2f}`")
                st.write(f"**Retriever:** `{retriever_label}`")

                st.markdown("### 📑 Top Documents (RRF-ranked)")
                for idx, doc in enumerate(result.docs):
                    title = doc.metadata.get("title", "")
                    doc_id = doc.metadata.get("doc_id", "")
                    uuid_val = doc.metadata.get("uuid", "")
                    source = doc.metadata.get("source", "")
                    rrf_score = result.rrf_scores.get(idx, 0.0)

                    st.markdown(f"**{idx+1})** {title or '(no title)'} — RRF: `{rrf_score:.4f}`")
                    if doc_id:
                        st.write(f"doc_id: {doc_id}")
                    if uuid_val:
                        st.write(f"uuid: {uuid_val}")
                    if source:
                        st.write(f"source: {source}")

                    preview = (doc.page_content or "").strip()
                    if len(preview) > 500:
                        preview = preview[:500] + " …"
                    st.code(preview)
                    st.divider()

        st.session_state.messages.append({"role": "assistant", "content": full_response})