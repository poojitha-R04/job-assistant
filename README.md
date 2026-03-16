# 🎯 JobSense AI - Smart Job Application Assistant

> An intelligent AI-powered job application assistant built using **Endee Vector Database**, Google Gemini AI, and Streamlit.

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![Endee](https://img.shields.io/badge/Vector%20DB-Endee-purple)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-red)
![Gemini](https://img.shields.io/badge/LLM-Google%20Gemini-green)

---

## 🌟 What is JobSense AI?

JobSense AI helps job seekers understand how well their resume matches a job description using the power of **semantic search** and **RAG (Retrieval Augmented Generation)**.

Simply upload your resume, paste a job description, and get:

- ✅ **Match Score** — How well your profile fits the job
- ✅ **Matching Skills** — What you already have
- ❌ **Missing Skills** — What gaps you need to fill
- 📝 **Improved Resume Summary** — AI rewrites your summary
- 💬 **Chat Interface** — Ask anything about your application

---

## 🏗️ System Design

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   User uploads  │────▶│  Sentence Trans  │────▶│  Endee Vector   │
│  Resume + JD    │     │  former Embedder │     │   Database      │
└─────────────────┘     └──────────────────┘     └────────┬────────┘
                                                           │
                                                    Semantic Search
                                                           │
┌─────────────────┐     ┌──────────────────┐     ┌────────▼────────┐
│   Streamlit UI  │◀────│  Google Gemini   │◀────│ Relevant Chunks │
│   (Results +    │     │  LLM Analysis    │     │   Retrieved     │
│    Chat)        │     └──────────────────┘     └─────────────────┘
└─────────────────┘
```

## 🗄️ How Endee Vector Database is Used

Endee is the **core** of this application:

1. **Resume Storage** — Resume text is chunked and stored as vectors in `resume_index`
2. **Job Description Storage** — JD is chunked and stored in `job_index`
3. **Semantic Search** — When user asks a question, Endee finds the most relevant chunks
4. **Match Scoring** — Cosine similarity between resume and JD vectors gives the match score

---

## 🛠️ Tech Stack

| Technology                | Purpose                                              |
| ------------------------- | ---------------------------------------------------- |
| **Endee**                 | Vector Database for storing and searching embeddings |
| **Python**                | Core programming language                            |
| **Streamlit**             | Web UI framework                                     |
| **Sentence Transformers** | Free local text embeddings (all-MiniLM-L6-v2)        |
| **Google Gemini**         | LLM for AI analysis and chat                         |
| **PyPDF2**                | PDF text extraction                                  |
| **Docker**                | Running Endee locally                                |

---

## ⚙️ Setup Instructions

### Prerequisites

- Python 3.8+
- Docker Desktop
- Google Gemini API Key (free at https://aistudio.google.com)

### Step 1: Clone the Repository

```bash
git clone https://github.com/poojitha-R04/job-assistant
cd job-assistant
```

### Step 2: Start Endee Vector Database

```bash
docker compose up -d
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Run the App

```bash
streamlit run app.py
```

### Step 5: Open in Browser

```
http://localhost:8501
```

---

## 🚀 How to Use

1. Enter your **Gemini API Key** in the sidebar
2. Upload your **Resume as PDF**
3. Paste the **Job Description**
4. Click **"Analyze My Application"**
5. View your **Match Score + AI Insights**
6. Use the **Chat** to ask follow-up questions

---

## 👩‍💻 Author

**Poojitha R**

- GitHub: [@poojitha-R04](https://github.com/poojitha-R04)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
