ქვემოთ გაქვს **საბოლოო, პროფესიონალური `README.md`** ქართულად — შენს პროექტზე მორგებული, სრული flow-ით, ფაილების აღწერით, ინსტალაციით, გამოყენებით, არქიტექტურით და ტექნიკური დეტალებით.

შეგიძლია პირდაპირ დააკოპირო 👉 `README.md`

---

# 🏛️ RS InfoHub RAG AI Agent

**RS InfoHub RAG** არის Retrieval-Augmented Generation (RAG) AI აგენტი, რომელიც ეძებს და პასუხობს საქართველოს შემოსავლების სამსახურის **InfoHub** პლატფორმაზე განთავსებულ დოკუმენტებზე დაყრდნობით.

აგენტს შეუძლია:

* 📄 10,000+ დოკუმენტის ინდექსაცია
* 🔎 Hybrid ძიება (BM25 + Semantic Vector Search)
* 🇬🇪 პასუხი ქართულად
* 🔗 შესაბამისი წყაროების ავტომატური ჩვენება
* 🧪 Debug რეჟიმი retrieval-ის ანალიზისთვის

---

# ⚙️ ტექნოლოგიები

| კომპონენტი     | ტექნოლოგია            |
| -------------- | --------------------- |
| UI             | Streamlit             |
| LLM            | Gemini 2.5 Flash Lite |
| Embeddings     | Gemini Embedding 001  |
| Vector DB      | ChromaDB              |
| Keyword Search | BM25 (rank_bm25)      |
| Parsing        | BeautifulSoup         |
| Language       | Python 3.10+          |

---

# 📁 ფაილების სტრუქტურა

```
RS-INFOHUB-RAG
│
├── data/
│   ├── chroma_db/        # Vector database (Chroma persistence)
│   ├── raw_docs.json     # ჩამოტვირთული InfoHub დოკუმენტები
│   ├── chunks.jsonl      # დაჭრილი ტექსტი BM25 + vector search-ისთვის
│   ├── api_sample.json   # API debug sample
│   └── processed.txt     # დამუშავების ლოგი (optional)
│
├── src/
│   ├── app.py            # მთავარი Streamlit RAG აპლიკაცია
│   ├── download_data.py  # InfoHub API-დან დოკუმენტების ჩამოტვირთვა
│   ├── ingest.py         # ტექსტის დაჭრა + ChromaDB ინდექსაცია
│   ├── debug_index.py    # ინდექსის დიაგნოსტიკა (optional)
│   └── crawl_links.py    # UUID ლინკების გენერაცია (optional)
│
├── .env                  # API keys
├── requirements.txt     # Python dependencies
└── README.md
```

---

# 🔄 სრული სამუშაო Flow

## 1️⃣ დოკუმენტების ჩამოტვირთვა

```bash
python src/download_data.py
```

📥 რა ხდება:

* InfoHub API-დან იწერება ყველა დოკუმენტი
* ინახება → `data/raw_docs.json`

---

## 2️⃣ ინდექსაცია (Chunking + Embeddings + ChromaDB)

```bash
python src/ingest.py
```

📊 რა ხდება:

1. HTML → სუფთა ტექსტი
2. ტექსტის დაჭრა (chunk_size=1000, overlap=200)
3. იქმნება `data/chunks.jsonl` → BM25-სთვის
4. იქმნება embeddings → ChromaDB-ში
5. ძველი ინდექსი იშლება და ახლიდან იქმნება

---

## 3️⃣ აპლიკაციის გაშვება

```bash
streamlit run src/app.py
```

🌐 გახსენი ბრაუზერში:

```
http://localhost:8501
```

---

# 🔎 Retrieval არქიტექტურა

გამოიყენება **Hybrid Search**:

### BM25 (Keyword Search)

* მუშაობს `chunks.jsonl`-ზე
* საუკეთესოა:

  * ბრძანება N
  * მუხლი
  * რიცხვები
  * ზუსტი ტერმინები

### Semantic Vector Search

* ChromaDB + Gemini Embeddings
* საუკეთესოა:

  * კონცეპტუალური კითხვები
  * ბუნებრივი ენა

### Hybrid Merge

BM25 + Vector შედეგები → merge → dedupe → Top-K

---

# 🧠 RAG Pipeline

```
User Query
   ↓
Hybrid Retrieval (BM25 + Vector)
   ↓
Top-K Context
   ↓
Prompt Template
   ↓
Gemini LLM
   ↓
ქართული პასუხი + წყაროები
```

---

# 🧪 Debug რეჟიმი

Sidebar-ში:

* 🔘 Debug რეჟიმი ON → აჩვენებს:

  * Vector score
  * Top-K დოკუმენტები
  * metadata (doc_id, uuid, source)
  * context preview

* 🔘 Debug OFF → Debug ქრება, **წყაროები რჩება**

---

# 🎚️ Sidebar პარამეტრები

| პარამეტრი  | აღწერა                                   |
| ---------- | ---------------------------------------- |
| Top-K      | რამდენი დოკუმენტი მოვიტანოთ retrieval-ზე |
| Threshold  | მინ. შესაბამისობა წყაროების საჩვენებლად  |
| Debug Mode | retrieval-ის დიაგნოსტიკა                 |

---

# 🗂️ გარემოს ცვლადები (.env)

შექმენი `.env`:

```
GOOGLE_API_KEY=your_api_key_here
```

---

# 📦 ინსტალაცია

```bash
python -m venv .venv
.\.venv\Scripts\activate

pip install -r requirements.txt
```

აუცილებელი პაკეტები:

```
streamlit
langchain
langchain-community
langchain-google-genai
langchain-chroma
rank_bm25
beautifulsoup4
lxml
python-dotenv
```

---

# 📊 მონაცემების მოცულობა

* 📄 ~10,000+ დოკუმენტი
* ✂️ ~34,000+ chunk
* 🧠 Hybrid retrieval ready

---

# ❗ როგორ მუშაობს წყაროები

წყაროს ლინკები გამოჩნდება მხოლოდ მაშინ, როცა:

```
vector_score >= threshold
```

ეს იცავს სისტემას არარელევანტური ლინკების ჩვენებისგან.

---

# 💬 მხარდაჭერილი კითხვები

### ✅ კარგი კითხვები

* „რას ამბობს ბრძანება N269?“
* „დღგ-ს გადახდის ვადები“
* „საბაჟო დეკლარაციის შევსება“
* „იმპორტის გადასახადები“

### ❌ არარელევანტური

* „გამარჯობა“
* „მადლობა“

ამ შემთხვევაში retrieval არ ხდება.

---

# 🧱 არქიტექტურა (High-Level)

```
InfoHub API
   ↓
download_data.py
   ↓
raw_docs.json
   ↓
ingest.py
   ├─ chunks.jsonl → BM25
   └─ ChromaDB → Vector
        ↓
app.py (Hybrid RAG)
        ↓
Streamlit UI
```

---

# 🚀 Deployment იდეები

შესაძლებელია გაშვება:

* Streamlit Cloud
* Docker
* VPS + Nginx

---

# 🛠️ Troubleshooting

### BM25 error

```
ImportError: rank_bm25
```

➡️

```bash
pip install rank_bm25
```

### DB not found

➡️ ჯერ გაუშვი:

```bash
python src/ingest.py
```

### წყაროები არ ჩანს

➡️ შეამცირე Threshold sidebar-ში

---

# 📌 Best Practices

* Threshold = 0.50–0.60
* Top-K = 4–6
* Debug მხოლოდ ტესტზე გამოიყენე

---

# 👨‍💻 ავტორი

RS InfoHub RAG – AI Assistant
შემუშავებულია Retrieval-Augmented Generation არქიტექტურით.

---