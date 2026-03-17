# 🎯 HireIQ - AI Career Intelligence Platform

> An intelligent AI-powered career platform built using **Endee Vector Database**, Groq LLaMA AI, and Streamlit.

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![Endee](https://img.shields.io/badge/Vector%20DB-Endee-purple)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-red)
![Groq](https://img.shields.io/badge/LLM-Groq%20LLaMA-green)

---

## 🌟 What is HireIQ?

HireIQ is an intelligent career platform that helps job seekers get hired faster by analyzing their resume against job descriptions using the power of **semantic search** and **AI**.

Upload your resume, paste a job description, and get:

- 📊 **Skills Analytics** — Match score + skill gap analysis
- 📝 **AI Resume Builder** — ATS-optimized resume tailored to the job
- ⚡ **Strategic Action Plan** — 3 actionable steps to strengthen your application
- 🧠 **Interview Intelligence** — 10+ tailored questions with suggested answers
- 🚀 **30-Day Career Roadmap** — Daily tasks with mind maps for each day
- 🤝 **AI Career Intelligence Assistant** — Chat with AI about your profile

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
│   Streamlit UI  │◀────│  Groq LLaMA 3.3  │◀────│ Relevant Chunks │
│  (Results +     │     │  70B AI Analysis │     │   Retrieved     │
│    Chat)        │     └──────────────────┘     └─────────────────┘
└─────────────────┘
```

## 🗄️ How Endee Vector Database is Used

Endee is the **core** of this application:

1. **Resume Storage** — Resume text is chunked and stored as vectors in `resume_index`
2. **Job Description Storage** — JD is chunked and stored in `job_index`
3. **Semantic Search** — Finds most relevant chunks for context
4. **Match Scoring** — Cosine similarity between resume and JD vectors gives the match score

---

## 🛠️ Tech Stack

| Technology                | Purpose                                              |
| ------------------------- | ---------------------------------------------------- |
| **Endee**                 | Vector Database for storing and searching embeddings |
| **Python**                | Core programming language                            |
| **Streamlit**             | Web UI framework                                     |
| **Sentence Transformers** | Free local text embeddings (all-MiniLM-L6-v2)        |
| **Groq LLaMA 3.3 70B**    | LLM for AI analysis, resume building and chat        |
| **Plotly**                | Skills proficiency chart visualization               |
| **PyPDF2**                | PDF text extraction                                  |
| **Docker**                | Running Endee locally                                |

---

## ⚙️ Setup Instructions

### Prerequisites

- Python 3.8+
- Docker Desktop
- Groq API Key (free at https://console.groq.com)

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
python -m streamlit run app.py
```

### Step 5: Open in Browser

```
http://localhost:8501
```

---

## 🚀 How to Use

1. Enter your **Groq API Key** in the sidebar
2. Upload your **Resume as PDF**
3. Paste the **Job Description**
4. Click **"Analyze with HireIQ"**
5. View your **Match Score + AI Insights**
6. Use the **Chat** to ask follow-up questions

---

## 👩‍💻 Author

**Poojitha R**

- GitHub: [@poojitha-R04](https://github.com/poojitha-R04)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
