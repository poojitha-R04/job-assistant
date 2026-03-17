import streamlit as st
from groq import Groq
import io
import plotly.graph_objects as go
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from endee import Endee, Precision

for _k, _v in {
    "analyzed": False, "score": 0, "resume_text": "",
    "job_text": "", "matching_skills": [], "missing_skills": [],
    "tips": [], "resume_built": "", "qa_text": "",
    "weeks_data": {}, "extra_qa": "", "messages": [],
    "groq_client": None,
}.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

st.set_page_config(page_title="HireIQ", page_icon="🎯", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
* { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; }

.hero {
    position: relative; overflow: hidden; background: #000;
    border-radius: 20px; padding: 3.5rem 2rem 2.5rem;
    text-align: center; margin-bottom: 24px; border: 1px solid #111;
}
.hero-grid {
    position: absolute; inset: 0;
    background-image: linear-gradient(rgba(255,255,255,0.04) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255,255,255,0.04) 1px, transparent 1px);
    background-size: 40px 40px;
}
.hero-glow {
    position: absolute; bottom: -50px; left: 50%; transform: translateX(-50%);
    width: 600px; height: 200px;
    background: radial-gradient(ellipse, rgba(124,58,237,0.45) 0%, rgba(56,189,248,0.2) 35%, rgba(16,185,129,0.1) 60%, transparent 80%);
    border-radius: 50%;
}
.hero-content { position: relative; z-index: 2; }
.hero-badge {
    display: inline-flex; align-items: center; gap: 6px;
    background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1);
    border-radius: 20px; padding: 5px 16px; font-size: 11px; color: #888;
    margin-bottom: 18px; transition: all 0.3s ease; cursor: default;
}
.hero-badge:hover { background: rgba(167,139,250,0.1); border-color: rgba(167,139,250,0.3); color: #a78bfa; transform: scale(1.05); }
.hero-badge-dot { width: 6px; height: 6px; border-radius: 50%; background: #a78bfa; }
.hero-title { font-size: 3.5rem; font-weight: 900; color: white; letter-spacing: -3px; line-height: 1; margin-bottom: 12px; }
.hero-title span { color: #a78bfa; }
.hero-sub { font-size: 1rem; color: #555; max-width: 480px; margin: 0 auto 24px; line-height: 1.7; }

.feat-grid { display: grid; grid-template-columns: repeat(4,1fr); gap: 10px; margin-top: 20px; }
.feat-card {
    background: #0a0a0a; border: 1px solid #1a1a1a;
    border-radius: 12px; padding: 14px 10px; text-align: center;
    transition: all 0.3s cubic-bezier(0.34,1.56,0.64,1); cursor: default;
}
.feat-card:hover { transform: scale(1.08) translateY(-4px); border-color: rgba(167,139,250,0.35); background: #0d0d12; box-shadow: 0 8px 30px rgba(124,58,237,0.2); }
.feat-icon { font-size: 1.5rem; margin-bottom: 6px; }
.feat-name { font-size: 11px; color: white; font-weight: 600; }
.feat-desc { font-size: 10px; color: #444; margin-top: 2px; }

.sh {
    font-size: 16px; font-weight: 800; color: white;
    letter-spacing: -0.5px; margin: 28px 0 8px;
    display: flex; align-items: center; gap: 10px;
    padding-bottom: 10px; border-bottom: 1px solid #1a1a1a;
}
.sh::after { content: ''; flex: 1; height: 1px; background: linear-gradient(90deg, #1a1a1a, transparent); }

.sh-sub { font-size: 13px; color: #666; margin: 0 0 8px; font-weight: 400; }
.sh-sub u { text-decoration: none; border-bottom: 1px solid rgba(167,139,250,0.4); color: #c4b5fd; padding-bottom: 1px; }
.sh-tags { display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 14px; }
.sh-tag { display: flex; align-items: center; gap: 5px; background: #0a0a0a; border: 1px solid #1a1a1a; border-radius: 6px; padding: 4px 10px; transition: all 0.2s ease; cursor: default; }
.sh-tag:hover { border-color: rgba(167,139,250,0.3); background: rgba(167,139,250,0.05); transform: scale(1.05); }
.sh-tag-icon { font-size: 11px; }
.sh-tag-text { font-size: 10px; color: #666; }

.card {
    background: #080808; border: 1px solid #151515;
    border-radius: 12px; padding: 14px; margin-bottom: 8px;
    transition: all 0.3s cubic-bezier(0.34,1.56,0.64,1);
}
.card:hover { transform: scale(1.02) translateY(-2px); border-color: rgba(167,139,250,0.25); box-shadow: 0 6px 24px rgba(0,0,0,0.5); background: #0a0a10; }
.card-row { display: flex; align-items: center; gap: 10px; }
.card-ic {
    width: 32px; height: 32px; border-radius: 8px;
    background: #111; border: 1px solid #1e1e1e;
    display: flex; align-items: center; justify-content: center;
    font-size: 14px; flex-shrink: 0; transition: all 0.3s ease;
}
.card:hover .card-ic { background: rgba(167,139,250,0.1); border-color: rgba(167,139,250,0.2); transform: scale(1.1); }
.card-title { font-size: 13px; color: #e8e8e8; font-weight: 600; line-height: 1.5; }
.card-desc { font-size: 12px; color: #777; margin-top: 4px; line-height: 1.7; }
.score-val { font-size: 2.5rem; font-weight: 900; color: white; text-align: center; letter-spacing: -2px; }
.score-lbl { font-size: 9px; color: #444; text-align: center; text-transform: uppercase; letter-spacing: 1px; margin-top: 4px; }

.tag { display: inline-block; background: #0d0d0d; color: #666; border: 1px solid #1a1a1a; border-radius: 6px; padding: 4px 12px; font-size: 12px; margin: 2px; transition: all 0.25s ease; cursor: default; }
.tag:hover { background: rgba(167,139,250,0.08); color: #a78bfa; border-color: rgba(167,139,250,0.2); transform: scale(1.08); }
.tag-green { background: rgba(16,185,129,0.08); color: #10b981; border-color: rgba(16,185,129,0.2); font-size: 12px; padding: 4px 12px; }
.tag-red { background: rgba(239,68,68,0.08); color: #ef4444; border-color: rgba(239,68,68,0.2); font-size: 12px; padding: 4px 12px; }

.resume-preview {
    background: white; border-radius: 12px; padding: 2.5rem; margin: 10px 0;
    color: #1a1a1a; font-family: 'Times New Roman', serif;
    box-shadow: 0 0 0 1px rgba(255,255,255,0.05); transition: all 0.3s ease;
}
.resume-preview:hover { transform: scale(1.005); box-shadow: 0 8px 40px rgba(0,0,0,0.6); }
.r-name { font-size: 1.6rem; font-weight: 700; text-align: center; margin-bottom: 4px; }
.r-contact { font-size: .85rem; text-align: center; color: #444; margin-bottom: 12px; }
.r-section { font-size: .9rem; font-weight: 700; letter-spacing: 1px; text-transform: uppercase; border-bottom: 1px solid #aaa; margin-top: 14px; margin-bottom: 4px; padding-bottom: 2px; }
.r-line { font-size: .85rem; margin: 2px 0; color: #222; }
.r-bullet { font-size: .85rem; margin: 2px 0 2px 16px; color: #222; }

.stButton>button {
    background: white !important; color: #000 !important;
    border: none !important; border-radius: 30px !important;
    font-size: 13px !important; font-weight: 700 !important;
    padding: 0.7rem 2rem !important;
    transition: all 0.3s cubic-bezier(0.34,1.56,0.64,1) !important;
    box-shadow: 0 2px 10px rgba(255,255,255,0.1) !important;
}
.stButton>button:hover { transform: scale(1.06) !important; box-shadow: 0 6px 20px rgba(255,255,255,0.15) !important; }
.stButton>button:active { transform: scale(0.97) !important; }

.stFileUploader>div { background: #080808 !important; border: 1px dashed #222 !important; border-radius: 12px !important; transition: all 0.3s ease !important; }
.stFileUploader>div:hover { border-color: rgba(167,139,250,0.4) !important; background: #0a0a12 !important; }

.stTextArea>div>textarea { background: #080808 !important; color: #ccc !important; border: 1px solid #1a1a1a !important; border-radius: 12px !important; transition: all 0.3s ease !important; }
.stTextArea>div>textarea:focus { border-color: rgba(167,139,250,0.4) !important; box-shadow: 0 0 0 2px rgba(167,139,250,0.1) !important; }

.streamlit-expanderHeader { background: #080808 !important; border: 1px solid #151515 !important; border-radius: 10px !important; color: #ccc !important; transition: all 0.3s ease !important; }
.streamlit-expanderHeader:hover { border-color: rgba(167,139,250,0.25) !important; background: #0a0a10 !important; transform: scale(1.01) !important; }

.mindmap-box { background: #050508; border: 1px solid #111; border-radius: 10px; padding: 1.5rem; font-family: monospace; color: #a78bfa; font-size: .85rem; white-space: pre-wrap; line-height: 1.8; transition: all 0.3s ease; }
.mindmap-box:hover { border-color: rgba(167,139,250,0.3); transform: scale(1.01); }

.answer-card { background: #050508; border-left: 2px solid #10b981; border-radius: 8px; padding: 1rem 1.4rem; margin: .3rem 0 .8rem 1rem; color: #6ee7b7; font-size: 12px; line-height: 1.8; transition: all 0.3s ease; }
.answer-card:hover { transform: scale(1.01); border-left-color: #34d399; }

::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #000; }
::-webkit-scrollbar-thumb { background: #222; border-radius: 2px; }
::-webkit-scrollbar-thumb:hover { background: #a78bfa; }

.main { background: #000 !important; }
.block-container { background: #000 !important; padding-top: 1rem !important; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
    <div class="hero-grid"></div>
    <div class="hero-glow"></div>
    <div class="hero-content">
        <div class="hero-badge">
            <span class="hero-badge-dot"></span>
            Powered by Endee Vector Database + Groq AI
        </div>
        <div class="hero-title">Hire<span>IQ</span></div>
        <div class="hero-sub">Intelligence That Gets You Hired.<br>Upload your resume, paste a job description, get hired faster.</div>
        <div class="feat-grid">
            <div class="feat-card"><div class="feat-icon">📊</div><div class="feat-name">Skills Analytics</div><div class="feat-desc">Gap analysis</div></div>
            <div class="feat-card"><div class="feat-icon">📝</div><div class="feat-name">Resume Builder</div><div class="feat-desc">Tailored to job</div></div>
            <div class="feat-card"><div class="feat-icon">🧠</div><div class="feat-name">Interview Prep</div><div class="feat-desc">10+ Q&A</div></div>
            <div class="feat-card"><div class="feat-icon">🚀</div><div class="feat-name">Career Roadmap</div><div class="feat-desc">30-day plan</div></div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_resource
def get_endee():
    c = Endee()
    c.set_base_url('http://localhost:8080/api/v1')
    return c

model   = load_model()
endee_c = get_endee()
RESUME_INDEX = "resume_index"
JOB_INDEX    = "job_index"
DIMENSION    = 384

def setup_endee():
    try:
        existing = [i.name for i in endee_c.list_indexes()]
    except:
        existing = []
    for name in [RESUME_INDEX, JOB_INDEX]:
        try:
            if name not in existing:
                endee_c.create_index(name=name, dimension=DIMENSION, space_type="cosine", precision=Precision.INT8)
        except:
            pass

def extract_text(pdf_file):
    data = pdf_file.read()
    reader = PdfReader(io.BytesIO(data))
    return "".join(p.extract_text() or "" for p in reader.pages)

def store_endee(text, index_name):
    words  = text.split()
    chunks = [" ".join(words[i:i+300]) for i in range(0,len(words),300)] or [text[:500]]
    idx    = endee_c.get_index(name=index_name)
    for i, chunk in enumerate(chunks):
        vec = model.encode(chunk).tolist()
        try:
            idx.upsert([{"id":f"c{i}","vector":vec,"meta":{"text":chunk}}])
        except:
            pass

def calc_score(rt, jt):
    r = model.encode(rt[:1000])
    j = model.encode(jt[:1000])
    sim = float(sum(a*b for a,b in zip(r,j)) / (sum(a**2 for a in r)**.5 * sum(b**2 for b in j)**.5))
    return round((sim+1)/2*100, 2)

def groq_ask(prompt):
    r = st.session_state.groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role":"user","content":prompt}],
        max_tokens=2000
    )
    return r.choices[0].message.content

def skills_chart(matching, missing):
    skills = (matching+missing)[:12]
    scores = [round(95-i*3,1) if s in matching else round(45-i*3,1) for i,s in enumerate(skills)]
    colors = ['#10b981' if s in matching else '#ef4444' for s in skills]
    fig = go.Figure(go.Bar(
        x=skills, y=scores, marker_color=colors, opacity=.85,
        text=[f"{v}%" for v in scores], textposition='outside',
        textfont=dict(color='white',size=11), width=.6
    ))
    fig.add_hline(y=70, line_dash="dash", line_color="rgba(255,255,255,.2)",
                  annotation_text="Proficiency Threshold", annotation_font_color="#666")
    fig.update_layout(
        title=dict(text="Skill Proficiency Index", font=dict(color='white',size=14), x=.5),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='#080808',
        font=dict(color='white',family='Inter'),
        xaxis=dict(showgrid=False,tickangle=-30,tickfont=dict(size=10),color='#444'),
        yaxis=dict(showgrid=True,gridcolor='#111',range=[0,115],title="Proficiency Score (%)",color='#444'),
        margin=dict(l=20,r=20,t=50,b=80), height=400, showlegend=False, bargap=.3
    )
    return fig

def render_resume(text):
    lines = text.split('\n')
    html  = []
    SECTIONS = {'SKILLS','PROJECTS','INTERNSHIP','ACHIEVEMENTS','EDUCATION','EXPERIENCE','SUMMARY','CERTIFICATIONS','TECHNICAL'}
    skip = 0
    for line in lines:
        cl = line.strip().replace('**','')
        if not cl:
            html.append('<div style="height:6px"></div>')
            continue
        if skip < 2:
            if skip == 0:
                html.append(f'<div class="r-name">{cl}</div>')
            else:
                html.append(f'<div class="r-contact">{cl}</div>')
            skip += 1
            continue
        upper = cl.upper()
        if any(upper.startswith(s) for s in SECTIONS):
            html.append(f'<div class="r-section">{cl}</div>')
        elif cl.startswith('–') or cl.startswith('-') or cl.startswith('•'):
            html.append(f'<div class="r-bullet">{cl}</div>')
        elif ':' in cl and len(cl) < 80:
            parts = cl.split(':',1)
            html.append(f'<div class="r-line"><b>{parts[0]}:</b>{parts[1]}</div>')
        else:
            html.append(f'<div class="r-line">{cl}</div>')
    st.markdown(f'<div class="resume-preview">{"".join(html)}</div>', unsafe_allow_html=True)

def render_qa(text):
    lines = text.split('\n')
    i = 0
    while i < len(lines):
        ln = lines[i].strip()
        if ln and ln[0]=='Q' and len(ln)>2 and ln[1].isdigit() and ':' in ln:
            q = ln.split(':',1)[1].strip()
            a = ""
            if i+1 < len(lines):
                nx = lines[i+1].strip()
                if nx and nx[0]=='A' and ':' in nx:
                    a = nx.split(':',1)[1].strip()
                    i += 1
            st.markdown(f'<div class="card"><div class="card-row"><div class="card-ic">🎯</div><div><div class="card-title">{ln.split(":")[0]}: {q}</div></div></div></div>', unsafe_allow_html=True)
            if a:
                st.markdown(f'<div class="answer-card">💡 <b>Suggested Answer:</b> {a}</div>', unsafe_allow_html=True)
        i += 1

def parse_roadmap(text):
    weeks, cur = {}, None
    for ln in text.split('\n'):
        ln = ln.strip()
        if not ln: continue
        if ln.startswith('WEEK'):
            cur = ln; weeks[cur] = []
        elif ln.startswith('Day') and cur:
            weeks[cur].append(ln)
    return weeks

def mindmap(day_label, day_task):
    return groq_ask(f"""Create a visual text-based mind map for: {day_task}
Format EXACTLY:
                    🎯 [{day_label}]
                          |
        __________________|__________________
        |                 |                 |
   [TOPIC 1]         [TOPIC 2]         [TOPIC 3]
        |                 |                 |
  ┌─────┴─────┐     ┌─────┴─────┐     ┌─────┴─────┐
  │           │     │           │     │           │
[step 1]  [step 2] [step 1] [step 2] [step 1] [step 2]
Use relevant topics. Keep it clean and professional.""")

with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:1.5rem 0 1rem">
        <div style="font-size:2rem;font-weight:900;color:white;letter-spacing:-2px;line-height:1">
            Hire<span style="color:#a78bfa">IQ</span>
        </div>
        <div style="font-size:11px;color:#555;margin-top:6px;text-transform:uppercase;letter-spacing:2px">Career Intelligence</div>
        <div style="margin-top:12px;height:1px;background:linear-gradient(90deg,transparent,#333,transparent)"></div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<div style="font-size:11px;color:#888;text-transform:uppercase;letter-spacing:1.5px;margin-bottom:8px;font-weight:700">API Configuration</div>', unsafe_allow_html=True)
    api_key = st.text_input("Groq API Key", type="password", placeholder="gsk_...")
    st.markdown("""<div style="height:1px;background:linear-gradient(90deg,transparent,#222,transparent);margin:16px 0"></div>
    <div style="font-size:11px;color:#888;text-transform:uppercase;letter-spacing:1.5px;margin-bottom:10px;font-weight:700">How to use</div>""", unsafe_allow_html=True)
    for i,s in enumerate(["Enter Groq API key","Upload Resume PDF","Paste Job Description","Click Analyze!"],1):
        st.markdown(f'''<div style="display:flex;align-items:center;gap:10px;padding:8px 0;border-bottom:1px solid #0d0d0d">
            <div style="width:22px;height:22px;border-radius:6px;background:rgba(167,139,250,0.1);border:1px solid rgba(167,139,250,0.2);
                display:flex;align-items:center;justify-content:center;font-size:11px;color:#a78bfa;font-weight:700;flex-shrink:0">{i}</div>
            <div style="font-size:12px;color:#888">{s}</div>
        </div>''', unsafe_allow_html=True)
    st.markdown("""<div style="height:1px;background:linear-gradient(90deg,transparent,#222,transparent);margin:16px 0"></div>
    <div style="font-size:11px;color:#888;text-transform:uppercase;letter-spacing:1.5px;margin-bottom:10px;font-weight:700">Powered by</div>""", unsafe_allow_html=True)
    for icon,name in [("🗄️","Endee Vector Database"),("🤖","Groq LLaMA 3.3 70B"),("🔍","Sentence Transformers"),("📊","Plotly Visualizations")]:
        st.markdown(f'''<div style="display:flex;align-items:center;gap:10px;padding:8px 0;border-bottom:1px solid #0d0d0d">
            <div style="font-size:14px">{icon}</div>
            <div style="font-size:12px;color:#777;font-weight:500">{name}</div>
        </div>''', unsafe_allow_html=True)

c1, c2 = st.columns(2)
with c1:
    st.markdown('<div class="sh">📄 Upload Resume</div>', unsafe_allow_html=True)
    resume_file = st.file_uploader("Resume PDF", type="pdf", label_visibility="collapsed")
    if resume_file:
        st.success(f"✅ {resume_file.name} uploaded!")
with c2:
    st.markdown('<div class="sh">💼 Job Description</div>', unsafe_allow_html=True)
    job_description = st.text_area("JD", height=130, label_visibility="collapsed",
                                   placeholder="Paste the full job description here...")

st.markdown("<br>", unsafe_allow_html=True)

if st.button("🎯 Analyze with HireIQ", use_container_width=True):
    if not api_key:
        st.error("❌ Please enter your Groq API key!")
    elif not resume_file:
        st.error("❌ Please upload your resume!")
    elif not job_description:
        st.error("❌ Please paste the job description!")
    else:
        with st.spinner("🔄 HireIQ is analyzing your profile..."):
            try:
                st.session_state.groq_client = Groq(api_key=api_key)
                setup_endee()
                rt = extract_text(resume_file)
                store_endee(rt, RESUME_INDEX)
                store_endee(job_description, JOB_INDEX)
                st.session_state.resume_text = rt
                st.session_state.job_text    = job_description
                st.session_state.score       = calc_score(rt, job_description)

                raw = groq_ask(f"""Analyze resume vs job.
RESUME:{rt[:2000]}
JOB:{job_description[:1000]}
Return EXACTLY:
MATCHING_SKILLS: s1,s2,s3,s4,s5
MISSING_SKILLS: s1,s2,s3,s4,s5
TIPS: t1 | t2 | t3""")
                ms,gs,tips = [],[],[]
                for ln in raw.split('\n'):
                    if 'MATCHING_SKILLS:' in ln:
                        ms = [x.strip() for x in ln.split(':',1)[1].split(',') if x.strip()]
                    elif 'MISSING_SKILLS:' in ln:
                        gs = [x.strip() for x in ln.split(':',1)[1].split(',') if x.strip()]
                    elif 'TIPS:' in ln:
                        tips = [x.strip() for x in ln.split(':',1)[1].split('|') if x.strip()]
                st.session_state.matching_skills = ms
                st.session_state.missing_skills  = gs
                st.session_state.tips            = tips

                st.session_state.resume_built = groq_ask(f"""
You are an expert resume writer. BUILD a NEW tailored resume — do NOT copy paste the old one.
CANDIDATE BACKGROUND: {rt[:2500]}
TARGET JOB: {job_description[:800]}
INSTRUCTIONS:
- Extract candidate real name, contact, education, projects, internship
- REWRITE every bullet point using keywords from the job description
- REORDER skills putting most relevant ones first
- Make every sentence sound written FOR this specific job
- Do NOT copy paste original resume bullets
FORMAT — clean, no asterisks, no markdown:
[Full Name]
[Phone] | [Email] | [LinkedIn] | GitHub: [value] | Portfolio: [value]
SKILLS
Programming: [most relevant first]
Frontend: [tailored to job]
Backend: [tailored to job]
Database: [tailored to job]
Core Concepts: [tailored to job]
PROJECTS
[Project Name]                                     [Date]
[Tech stack]
– [Rewritten bullet with job keywords]
– [Rewritten bullet]
– [Rewritten bullet]
INTERNSHIP
[Role]                                             [Date]
[Company]
– [Rewritten bullet with job keywords]
– [Rewritten bullet]
ACHIEVEMENTS
– [Achievement]
– [Achievement]
EDUCATION
[College]                                          [Year]
[Degree]                                           [Location]""")

                st.session_state.qa_text = groq_ask(f"""Generate exactly 10 interview questions WITH detailed answers.
JOB:{job_description[:800]}
RESUME:{rt[:800]}
Format EXACTLY:
Q1: question
A1: detailed answer
Q2: question
A2: detailed answer
Continue until Q10.""")

                st.session_state.weeks_data = parse_roadmap(groq_ask(f"""
Create a detailed 30-day career roadmap.
Missing skills: {', '.join(gs)}
Job: {job_description[:400]}
Format EXACTLY:
WEEK 1:
Day 1: task
Day 2: task
Day 3: task
Day 4: task
Day 5: task
Day 6: task
Day 7: task
WEEK 2:
Day 8: task
Day 9: task
Day 10: task
Day 11: task
Day 12: task
Day 13: task
Day 14: task
WEEK 3:
Day 15: task
Day 16: task
Day 17: task
Day 18: task
Day 19: task
Day 20: task
Day 21: task
WEEK 4:
Day 22: task
Day 23: task
Day 24: task
Day 25: task
Day 26: task
Day 27: task
Day 28-30: Final review and mock interviews"""))

                st.session_state.analyzed = True
                st.session_state.extra_qa = ""
                st.balloons()

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

if st.session_state.analyzed:
    sc = st.session_state.score
    ms = st.session_state.matching_skills
    gs = st.session_state.missing_skills
    rt = st.session_state.resume_text

    # Dashboard
    st.markdown('<div class="sh">📊 Candidate Evaluation Dashboard</div>', unsafe_allow_html=True)
    st.markdown('''<div class="sh-sub">Your profile <u>analyzed</u> against the <u>job description</u></div>
    <div class="sh-tags">
        <div class="sh-tag"><span class="sh-tag-icon">📊</span><span class="sh-tag-text">AI Powered</span></div>
        <div class="sh-tag"><span class="sh-tag-icon">⚡</span><span class="sh-tag-text">Real Time</span></div>
        <div class="sh-tag"><span class="sh-tag-icon">🎯</span><span class="sh-tag-text">Job Tailored</span></div>
    </div>''', unsafe_allow_html=True)
    ca,cb,cc = st.columns(3)
    with ca:
        badge = "🟢 Strong Fit" if sc>70 else "🟡 Moderate Fit" if sc>50 else "🔴 Skill Gap"
        st.markdown(f'''<div class="card" style="text-align:center">
            <div class="score-val">{sc}%</div>
            <div class="score-lbl">Overall Fit Score</div>
            <div style="margin-top:8px;font-size:12px;color:#888">{badge}</div>
        </div>''', unsafe_allow_html=True)
    with cb:
        st.markdown(f'''<div class="card" style="text-align:center">
            <div style="font-size:2.5rem;font-weight:900;color:white;letter-spacing:-2px">{max(1,len(rt)//500)}</div>
            <div class="score-lbl">Pages Analyzed</div>
            <div style="margin-top:10px;font-size:2rem;font-weight:800;color:#a78bfa;letter-spacing:-1px">{len(rt.split())}</div>
            <div class="score-lbl">Words Processed</div>
        </div>''', unsafe_allow_html=True)
    with cc:
        st.markdown(f'''<div class="card" style="text-align:center">
            <div style="font-size:2.5rem;font-weight:900;color:#10b981;letter-spacing:-2px">{len(ms)}</div>
            <div class="score-lbl">Matching Skills</div>
            <div style="margin-top:10px;font-size:2rem;font-weight:800;color:#ef4444;letter-spacing:-1px">{len(gs)}</div>
            <div class="score-lbl">Skill Gaps</div>
        </div>''', unsafe_allow_html=True)

    # Skills Analytics
    st.markdown('<div class="sh">📊 Skills Analytics</div>', unsafe_allow_html=True)
    st.markdown('''<div class="sh-sub">Skills <u>matched and gaps</u> identified from your <u>resume vs job description</u></div>
    <div class="sh-tags">
        <div class="sh-tag"><span class="sh-tag-icon">✓</span><span class="sh-tag-text">AI Powered</span></div>
        <div class="sh-tag"><span class="sh-tag-icon">📊</span><span class="sh-tag-text">Gap Analysis</span></div>
        <div class="sh-tag"><span class="sh-tag-icon">⚡</span><span class="sh-tag-text">Real Time</span></div>
        <div class="sh-tag"><span class="sh-tag-icon">🎯</span><span class="sh-tag-text">Job Tailored</span></div>
    </div>''', unsafe_allow_html=True)
    cs1,cs2 = st.columns(2)
    with cs1:
        st.markdown('<div style="font-size:12px;color:#777;margin-bottom:8px;font-weight:600">✅ Acquired Skills</div>', unsafe_allow_html=True)
        tags = "".join(f'<span class="tag tag-green">✓ {s}</span>' for s in ms)
        st.markdown(f'<div class="card">{tags}</div>', unsafe_allow_html=True)
    with cs2:
        st.markdown('<div style="font-size:12px;color:#777;margin-bottom:8px;font-weight:600">⚠️ Skill Gaps</div>', unsafe_allow_html=True)
        tags = "".join(f'<span class="tag tag-red">✗ {s}</span>' for s in gs)
        st.markdown(f'<div class="card">{tags}</div>', unsafe_allow_html=True)
    if ms or gs:
        st.plotly_chart(skills_chart(ms,gs), use_container_width=True)

    # Resume Builder
    st.markdown('<div class="sh">📝 AI Resume Builder</div>', unsafe_allow_html=True)
    st.markdown('''<div class="sh-sub">AI-crafted resume <u>tailored specifically</u> for this <u>job description</u></div>
    <div class="sh-tags">
        <div class="sh-tag"><span class="sh-tag-icon">🤖</span><span class="sh-tag-text">AI Generated</span></div>
        <div class="sh-tag"><span class="sh-tag-icon">📝</span><span class="sh-tag-text">Job Tailored</span></div>
        <div class="sh-tag"><span class="sh-tag-icon">✨</span><span class="sh-tag-text">ATS Optimized</span></div>
        <div class="sh-tag"><span class="sh-tag-icon">🎯</span><span class="sh-tag-text">Keyword Rich</span></div>
    </div>''', unsafe_allow_html=True)
    render_resume(st.session_state.resume_built)

    # Strategic Action Plan
    if st.session_state.tips:
        st.markdown('<div class="sh">⚡ Strategic Action Plan</div>', unsafe_allow_html=True)
        st.markdown('''<div class="sh-sub">Actionable steps to <u>strengthen</u> your <u>application</u></div>
        <div class="sh-tags">
            <div class="sh-tag"><span class="sh-tag-icon">⚡</span><span class="sh-tag-text">3 Actions</span></div>
            <div class="sh-tag"><span class="sh-tag-icon">🎯</span><span class="sh-tag-text">Job Focused</span></div>
            <div class="sh-tag"><span class="sh-tag-icon">✅</span><span class="sh-tag-text">Actionable</span></div>
        </div>''', unsafe_allow_html=True)
        for i,tip in enumerate(st.session_state.tips,1):
            st.markdown(f'<div class="card"><div class="card-row"><div class="card-ic">⚡</div><div><div class="card-title">Action {i}</div><div class="card-desc">{tip}</div></div></div></div>', unsafe_allow_html=True)

    # Interview Questions
    st.markdown('<div class="sh">🧠 Interview Questions</div>', unsafe_allow_html=True)
    st.markdown('''<div class="sh-sub"><u>10 tailored questions</u> with suggested answers based on the <u>job description</u></div>
    <div class="sh-tags">
        <div class="sh-tag"><span class="sh-tag-icon">🧠</span><span class="sh-tag-text">10 Questions</span></div>
        <div class="sh-tag"><span class="sh-tag-icon">💡</span><span class="sh-tag-text">With Answers</span></div>
        <div class="sh-tag"><span class="sh-tag-icon">🎯</span><span class="sh-tag-text">Job Tailored</span></div>
    </div>''', unsafe_allow_html=True)
    render_qa(st.session_state.qa_text)

    # Generate Additional Questions
    st.markdown('<div class="sh">➕ Generate Additional Interview Questions</div>', unsafe_allow_html=True)
    st.markdown('''<div class="sh-sub">Need more practice? <u>Generate</u> as many questions as you <u>want</u></div>
    <div class="sh-tags">
        <div class="sh-tag"><span class="sh-tag-icon">➕</span><span class="sh-tag-text">Custom Count</span></div>
        <div class="sh-tag"><span class="sh-tag-icon">💡</span><span class="sh-tag-text">With Answers</span></div>
        <div class="sh-tag"><span class="sh-tag-icon">🎯</span><span class="sh-tag-text">Job Tailored</span></div>
    </div>''', unsafe_allow_html=True)
    extra_num = st.slider("How many additional questions?", min_value=1, max_value=20, value=5, key="extra_slider")
    if st.button("🎯 Generate More Questions", use_container_width=True, key="gen_more"):
        with st.spinner("Generating..."):
            try:
                st.session_state.extra_qa = groq_ask(f"""Generate exactly {extra_num} interview questions WITH detailed answers.
JOB:{st.session_state.job_text[:800]}
Format EXACTLY:
Q1: question
A1: detailed answer
Continue until Q{extra_num}.""")
            except Exception as e:
                st.error(f"❌ {str(e)}")
    if st.session_state.extra_qa:
        render_qa(st.session_state.extra_qa)

    # 30-Day Roadmap
    st.markdown('<div class="sh">🚀 30-Day Career Roadmap</div>', unsafe_allow_html=True)
    st.markdown('''<div class="sh-sub">Day-by-day plan to <u>bridge skill gaps</u> and <u>land the job</u></div>
    <div class="sh-tags">
        <div class="sh-tag"><span class="sh-tag-icon">🚀</span><span class="sh-tag-text">30 Days</span></div>
        <div class="sh-tag"><span class="sh-tag-icon">📅</span><span class="sh-tag-text">Daily Tasks</span></div>
        <div class="sh-tag"><span class="sh-tag-icon">🧠</span><span class="sh-tag-text">Mind Maps</span></div>
        <div class="sh-tag"><span class="sh-tag-icon">🎯</span><span class="sh-tag-text">Skill Focused</span></div>
    </div>''', unsafe_allow_html=True)
    WEEK_COLORS = ['#a78bfa','#38bdf8','#10b981','#f59e0b']
    for wi,(wname,days) in enumerate(st.session_state.weeks_data.items()):
        wc = WEEK_COLORS[wi%4]
        with st.expander(f"📅 {wname} — click to expand", expanded=False):
            for day_line in days:
                if ':' in day_line:
                    dlabel,dtask = day_line.split(':',1)
                    dlabel,dtask = dlabel.strip(),dtask.strip()
                else:
                    dlabel,dtask = day_line.strip(),""
                with st.expander(f"📌 {dlabel}: {dtask[:55]}", expanded=False):
                    st.markdown(f'<div class="card"><div class="card-title">📋 Task</div><div class="card-desc">{dtask}</div></div>', unsafe_allow_html=True)
                    with st.expander("🧠 View Mind Map", expanded=False):
                        mk = f"mm_{wname}_{dlabel}"
                        if mk not in st.session_state:
                            st.session_state[mk] = mindmap(dlabel,dtask)
                        st.markdown(f'<div class="mindmap-box">{st.session_state[mk]}</div>', unsafe_allow_html=True)

    # AI Career Intelligence Assistant
    st.markdown('<div class="sh">🤝 AI Career Intelligence Assistant</div>', unsafe_allow_html=True)
    st.markdown('''<div class="sh-sub">Ask anything about your <u>resume</u>, <u>job fit</u>, or <u>career strategy</u></div>
    <div class="sh-tags">
        <div class="sh-tag"><span class="sh-tag-icon">🤝</span><span class="sh-tag-text">AI Powered</span></div>
        <div class="sh-tag"><span class="sh-tag-icon">💬</span><span class="sh-tag-text">24/7 Available</span></div>
        <div class="sh-tag"><span class="sh-tag-icon">🎯</span><span class="sh-tag-text">Career Focused</span></div>
    </div>''', unsafe_allow_html=True)
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    if user_input := st.chat_input("Ask your AI Career Intelligence Assistant..."):
        st.session_state.messages.append({"role":"user","content":user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
        with st.chat_message("assistant"):
            resp = groq_ask(f"Expert AI Career Intelligence Assistant. Resume:{st.session_state.resume_text[:600]} Job:{st.session_state.job_text[:300]} Question:{user_input} Give specific professional advice.")
            st.markdown(resp)
            st.session_state.messages.append({"role":"assistant","content":resp})

