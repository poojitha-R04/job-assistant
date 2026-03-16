import io
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from endee import Endee, Precision

model = SentenceTransformer('all-MiniLM-L6-v2')

client = Endee()
client.set_base_url('http://localhost:8080/api/v1')

RESUME_INDEX = "resume_index"
JOB_INDEX = "job_index"
DIMENSION = 384

def create_indexes():
    try:
        existing_names = [idx.name for idx in client.list_indexes()]
    except:
        existing_names = []
    try:
        if RESUME_INDEX not in existing_names:
            client.create_index(name=RESUME_INDEX, dimension=DIMENSION, space_type="cosine", precision=Precision.INT8)
    except:
        pass
    try:
        if JOB_INDEX not in existing_names:
            client.create_index(name=JOB_INDEX, dimension=DIMENSION, space_type="cosine", precision=Precision.INT8)
    except:
        pass

def extract_text_from_pdf(pdf_file):
    if hasattr(pdf_file, 'read'):
        pdf_bytes = pdf_file.read()
        reader = PdfReader(io.BytesIO(pdf_bytes))
    else:
        reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def chunk_text(text, chunk_size=300):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)
    if not chunks:
        chunks = [text[:500]]
    return chunks

def store_in_endee(text, index_name):
    chunks = chunk_text(text)
    index = client.get_index(name=index_name)
    for i, chunk in enumerate(chunks):
        embedding = model.encode(chunk).tolist()
        index.upsert([{
            "id": f"chunk_{i}",
            "vector": embedding,
            "meta": {"text": chunk}
        }])
    return chunks

def search_endee(query, index_name, top_k=5):
    try:
        index = client.get_index(name=index_name)
        query_vector = model.encode(query).tolist()
        results = index.query(vector=query_vector, top_k=top_k)
        if not results:
            return []
        texts = []
        for item in results:
            try:
                meta = getattr(item, 'meta', None)
                if meta is None:
                    meta = item.__dict__.get('meta', {})
                if isinstance(meta, dict):
                    texts.append(meta.get("text", ""))
                else:
                    texts.append(str(meta))
            except:
                texts.append("")
        return texts
    except:
        return []

def compute_match_score(resume_text, job_text):
    resume_embedding = model.encode(resume_text[:1000])
    job_embedding = model.encode(job_text[:1000])
    similarity = float(
        sum(a*b for a, b in zip(resume_embedding, job_embedding)) /
        (sum(a**2 for a in resume_embedding)**0.5 *
         sum(b**2 for b in job_embedding)**0.5)
    )
    return round((similarity + 1) / 2 * 100, 2)





