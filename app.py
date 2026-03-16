import streamlit as st
import google.generativeai as genai
import io
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from endee import Endee, Precision

st.set_page_config(page_title="JobSense AI", page_icon="🎯", layout="wide")
st.markdown("""<style>
.main-header{background:linear-gradient(90deg,#667eea 0%,#764ba2 100%);padding:2rem;border-radius:10px;color:white;text-align:center;margin-bottom:2rem;}
.score-box{background:linear-gradient(135deg,#667eea,#764ba2);color:white;padding:1.5rem;border-radius:10px;text-align:center;font-size:2rem;font-weight:bold;}
.insight-box{background:#f8f9fa;border-left:4px solid #667eea;padding:1rem;border-radius:5px;margin:0.5rem 0;}
</style>""", unsafe_allow_html=True)
st.markdown('<div class="main-header"><h1>🎯 JobSense AI</h1><p>Smart Job Application Assistant powered by Endee Vector Database</p></div>', unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_resource
def get_client():
    c = Endee()
    c.set_base_url('http://localhost:8080/api/v1')
    return c

model = load_model()
client = get_client()
RESUME_INDEX = "resume_index"
JOB_INDEX = "job_index"
DIMENSION = 384

def setup_endee():
    try:
        existing = [idx.name for idx in client.list_indexes()]
    except:
        existing = []
    for name in [RESUME_INDEX, JOB_INDEX]:
        try:
            if name not in existing:
                client.create_index(name=name, dimension=DIMENSION, space_type="cosine", precision=Precision.INT8)
        except:
            pass

def extract_text(pdf_file):
    pdf_bytes = pdf_file.read()
    reader = PdfReader(io.BytesIO(pdf_bytes))
    return "".join([p.extract_text() or "" for p in reader.pages])

def store_in_endee(text, index_name):
    words = text.split()
    chunks = [" ".join(words[i:i+300]) for i in range(0, len(words), 300)] or [text[:500]]
    index = client.get_index(name=index_name)
    for i, chunk in enumerate(chunks):
        vec = model.encode(chunk).tolist()
        try:
            index.upsert([{"id": f"chunk_{i}", "vector": vec, "meta": {"text": chunk}}])
        except:
            pass

def match_score(resume_text, job_text):
    r = model.encode(resume_text[:1000])
    j = model.encode(job_text[:1000])
    sim = float(sum(a*b for a,b in zip(r,j))/(sum(a**2 for a in r)**0.5*sum(b**2 for b in j)**0.5))
    return round((sim+1)/2*100, 2)

with st.sidebar:
    st.header("⚙️ Configuration")
    api_key = st.text_input("Enter Gemini API Key", type="password")
    st.markdown("---")
    st.markdown("### 📖 How to use:")
    st.markdown("1. Enter your Gemini API key")
    st.markdown("2. Upload your Resume (PDF)")
    st.markdown("3. Paste the Job Description")
    st.markdown("4. Click Analyze!")
    st.markdown("---")
    st.markdown("**Powered by:** 🗄️ Endee Vector Database")

col1, col2 = st.columns(2)
with col1:
    st.header("📄 Upload Resume")
    resume_file = st.file_uploader("Upload your Resume (PDF)", type="pdf")
    if resume_file:
        st.success("✅ Resume uploaded successfully!")
with col2:
    st.header("💼 Job Description")
    job_description = st.text_area("Paste the Job Description here", height=200)

if st.button("🚀 Analyze My Application", use_container_width=True):
    if not api_key:
        st.error("❌ Please enter your Gemini API key!")
    elif not resume_file:
        st.error("❌ Please upload your resume!")
    elif not job_description:
        st.error("❌ Please paste the job description!")
    else:
        with st.spinner("🔄 Analyzing using Endee Vector Database..."):
            try:
                genai.configure(api_key=api_key)
                gemini = genai.GenerativeModel('gemini-2.0-flash')
                setup_endee()
                resume_text = extract_text(resume_file)
                store_in_endee(resume_text, RESUME_INDEX)
                store_in_endee(job_description, JOB_INDEX)
                score = match_score(resume_text, job_description)
                st.session_state.resume_text = resume_text
                st.session_state.job_text = job_description
                st.session_state.gemini = gemini
                st.session_state.analyzed = True
                st.markdown("## 📊 Analysis Results")
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.markdown(f'<div class="score-box">{score}%<br><small style="font-size:1rem">Match Score</small></div>', unsafe_allow_html=True)
                with col_b:
                    status = "🟢 Strong" if score > 70 else "🟡 Moderate" if score > 50 else "🔴 Weak"
                    st.metric("Match Status", status)
                with col_c:
                    st.metric("Pages Analyzed", max(1, len(resume_text)//500))
                st.markdown("### 🤖 AI Insights")
                ai_prompt = f"You are a career coach. Analyze this resume vs job description.\n\nRESUME:\n{resume_text[:2000]}\n\nJOB:\n{job_description[:1000]}\n\nProvide:\n1. Top 3 matching skills\n2. Top 3 missing skills\n3. Improved resume summary\n4. 3 actionable tips"
                response = gemini.generate_content(ai_prompt)
                st.markdown(f'<div class="insight-box">{response.text}</div>', unsafe_allow_html=True)
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

if st.session_state.get("analyzed"):
    st.markdown("---")
    st.header("💬 Chat with AI about your Application")
    if "messages" not in st.session_state:
        st.session_state.messages = []
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    if user_input := st.chat_input("Ask anything about your resume or job fit..."):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
        with st.chat_message("assistant"):
            chat_prompt = f"Career coach. Resume: {st.session_state.resume_text[:1000]} Job: {st.session_state.job_text[:500]} Question: {user_input} Answer helpfully."
            response = st.session_state.gemini.generate_content(chat_prompt)
            st.markdown(response.text)
            st.session_state.messages.append({"role": "assistant", "content": response.text})



