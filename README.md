***

# 🏛️ RS InfoHub AI Agent (RAG)

**RS InfoHub AI** არის ინტელექტუალური საძიებო სისტემა (RAG - Retrieval-Augmented Generation), რომელიც შექმნილია საქართველოს ფინანსთა სამინისტროს შემოსავლების სამსახურის (RS) საინფორმაციო ჰაბის (`infohub.rs.ge`) დოკუმენტების დასამუშავებლად.

აგენტი იყენებს **Advanced Hybrid Search** ტექნოლოგიას — BM25 + Semantic Vector Search + **Multi-Query Expansion** + **Reciprocal Rank Fusion (RRF)** — რათა მომხმარებლის კითხვებს გასცეს ზუსტი და სრულყოფილი პასუხი შესაბამისი წყაროებით.

## 🚀 ძირითადი ფუნქციონალი

*   **Multi-Query Expansion:** LLM აგენერირებს კითხვის 3 ვარიაციას, რაც მკვეთრად ზრდის მოძიების სიზუსტეს.
*   **Hybrid Retrieval + RRF Fusion:** აერთიანებს BM25 (keyword) და Vector (semantic) ძიებას Reciprocal Rank Fusion ალგორითმით.
*   **BM25 ინდექსის პერსისტენცია:** BM25 ინდექსი ინახება `pickle` ფორმატში — აპლიკაცია წამებში იტვირთება.
*   **ქართული ენის მხარდაჭერა:** სრულად ადაპტირებულია ქართულ ტექსტზე და კითხვა-პასუხზე.
*   **წყაროების ვალიდაცია:** პასუხს თან ერთვის წყაროს ლინკები Threshold-ის მიხედვით.
*   **Smart Filtering:** ფილტრავს Smalltalk-ს (მაგ: "გამარჯობა", "როგორ ხარ").
*   **Debug Mode:** აჩვენებს ძიების სქორებს, RRF რანკინგს, მეტადატას და კონტექსტს.
*   **Conversation-Aware:** ითვალისწინებს წინა დიალოგს follow-up კითხვებისთვის.

## 🛠️ ტექნოლოგიური სტეკი

| კომპონენტი | ტექნოლოგია |
| :--- | :--- |
| **LLM** | Google Gemini 2.0 Flash |
| **Embeddings** | Gemini Embedding 001 |
| **Vector DB** | ChromaDB (Local) |
| **Keyword Search** | BM25 (`rank_bm25`) |
| **Fusion** | Reciprocal Rank Fusion (RRF) |
| **Query Expansion** | Multi-Query via LLM |
| **UI Framework** | Streamlit |
| **Data Scraping** | Python `requests` + `BeautifulSoup` |
| **Serialization** | `dill` (BM25 Pickle) |

---

## 📂 პროექტის სტრუქტურა

```text
RS-INFOHUB-RAG
│
├── .venv/                    # Python ვირტუალური გარემო
├── data/
│   ├── chroma_db/            # ვექტორული მონაცემთა ბაზა
│   ├── chunks.jsonl          # ტექსტური ჩანკები (BM25 fallback)
│   ├── bm25_retriever.pkl    # BM25 ინდექსი (სწრაფი ჩატვირთვა)
│   ├── raw_docs.jsonl        # API-დან ჩამოტვირთული დოკუმენტები
│   └── state.json            # სკრეიპერის მიმდინარე სტატუსი
│
├── src/
│   ├── app.py                # Streamlit აპლიკაცია (UI & Logic)
│   ├── retriever.py          # Hybrid Search + RRF + Query Expansion
│   ├── download_data.py      # InfoHub API-დან მონაცემების ჩამოტვირთვა
│   └── ingest.py             # მონაცემების დამუშავება და ინდექსაცია
│
├── .env                      # გარემოს ცვლადები (API Keys)
├── .gitignore
├── README.md
└── requirements.txt
```

---

## ⚙️ ინსტალაცია და გაშვება

### 1. რეპოზიტორიის მომზადება

```bash
# ვირტუალური გარემოს შექმნა
python -m venv .venv

# გააქტიურება (Windows)
.\.venv\Scripts\activate

# გააქტიურება (macOS/Linux)
source .venv/bin/activate
```

### 2. ბიბლიოთეკების დაყენება

```bash
pip install -r requirements.txt
```

### 3. გარემოს ცვლადები (.env)

შექმენით ფაილი `.env` პროექტის ძირში:

```env
GOOGLE_API_KEY=თქვენი_api_გასაღები_აქ
```

---

## 🔄 მონაცემების განახლება (Data Pipeline)

### ნაბიჯი 1: მონაცემების ჩამოტვირთვა
```bash
python src/download_data.py
```
*სკრიპტი ინახავს პროგრესს `state.json`-ში. შეწყვეტის შემთხვევაში გაგრძელდება იქიდან, სადაც გაჩერდა.*

### ნაბიჯი 2: ინდექსაცია
ამუშავებს HTML ტექსტს, ჭრის ჩანკებად, ქმნის ChromaDB ვექტორებს და BM25 ინდექსს (Pickle).

```bash
python src/ingest.py
```

---

## ▶️ აპლიკაციის გაშვება

```bash
streamlit run src/app.py
```

აპლიკაცია გაიხსნება: `http://localhost:8501`

---

## 🧠 როგორ მუშაობს სისტემა?

```
მომხმარებლის კითხვა
        │
        ▼
┌─────────────────────┐
│  Multi-Query        │  LLM აგენერირებს კითხვის 3 ვარიაციას
│  Expansion          │  (სხვადასხვა ფორმულირება)
└─────────┬───────────┘
          │
    ┌─────┴─────┐
    ▼           ▼
┌────────┐ ┌─────────┐
│  BM25  │ │ Vector  │   თითოეული ვარიაცია ეძებს ორივე ინდექსში
│Keyword │ │Semantic │
└───┬────┘ └────┬────┘
    │           │
    └─────┬─────┘
          ▼
┌─────────────────────┐
│  Reciprocal Rank    │  RRF აერთიანებს და აწესრიგებს
│  Fusion (RRF)       │  ყველა შედეგს
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│  Deduplication      │  დუბლიკატების წაშლა
│  + Top-K Selection  │
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│  LLM (Gemini 2.0    │  კონტექსტის საფუძველზე
│  Flash) Answer      │  პასუხის გენერირება
└─────────────────────┘
```

---

## 🎛️ მართვის პანელი (Sidebar)

*   **Debug რეჟიმი:** ტექნიკური დეტალები — RRF Scores, დოკუმენტების ID-ები, ტექსტის ნაწყვეტები.
*   **Top-K დოკუმენტი:** რამდენი დოკუმენტი წამოიღოს ბაზიდან (რეკომენდებულია 8-12).
*   **Threshold:** მინიმალური მსგავსების ზღვარი წყაროების ჩვენებისთვის.

---

## 👨‍💻 ავტორი
შექმნილია InfoHub-ის მონაცემებზე დაყრდნობით, რათა გამარტივდეს საგადასახადო და საბაჟო ინფორმაციის მოძიება.