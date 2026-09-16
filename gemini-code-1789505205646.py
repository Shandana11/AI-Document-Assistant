import os
import tempfile
import streamlit as st
import pypdf
import docx
import fitz  # PyMuPDF
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from groq import Groq
import gdown

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="RAG Knowledge Assistant (Local & Google Drive)",
    page_icon="🤖",
    layout="wide"
)

# --- STYLING ---
st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    .stButton>button { width: 100%; border-radius: 6px; font-weight: 600; }
    .source-box {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        border-left: 4px solid #4f46e5;
        padding: 12px;
        margin-bottom: 10px;
        border-radius: 4px;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# --- INITIALIZE MODELS & CACHE ---
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedding_model = load_embedding_model()

# --- EXTRACTION FUNCTIONS ---
def extract_pdf(file_path, filename):
    chunks_meta = []
    try:
        doc = fitz.open(file_path)
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            if text.strip():
                chunks_meta.append({
                    "filename": filename,
                    "page": page_num + 1,
                    "text": text.strip()
                })
    except Exception as e:
        st.error(f"Error reading PDF {filename}: {e}")
    return chunks_meta

def extract_docx(file_path, filename):
    chunks_meta = []
    try:
        doc = docx.Document(file_path)
        full_text = []
        for para in doc.paragraphs:
            if para.text.strip():
                full_text.append(para.text.strip())
        text = "\n".join(full_text)
        if text:
            chunks_meta.append({
                "filename": filename,
                "page": 1,
                "text": text
            })
    except Exception as e:
        st.error(f"Error reading DOCX {filename}: {e}")
    return chunks_meta

def extract_txt_md(file_path, filename):
    chunks_meta = []
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        if text.strip():
            chunks_meta.append({
                "filename": filename,
                "page": 1,
                "text": text.strip()
            })
    except Exception as e:
        st.error(f"Error reading {filename}: {e}")
    return chunks_meta

def extract_document(file_path, filename):
    ext = filename.lower().split('.')[-1]
    if ext == 'pdf':
        return extract_pdf(file_path, filename)
    elif ext == 'docx':
        return extract_docx(file_path, filename)
    elif ext in ['txt', 'md']:
        return extract_txt_md(file_path, filename)
    return []

# --- CHUNKING FUNCTION ---
def chunk_text(extracted_docs, chunk_size=500, overlap=50):
    all_chunks = []
    for doc in extracted_docs:
        filename = doc["filename"]
        page = doc["page"]
        text = doc["text"]
        
        words = text.split()
        if not words:
            continue
            
        i = 0
        while i < len(words):
            chunk_words = words[i:i + chunk_size]
            chunk_str = " ".join(chunk_words)
            all_chunks.append({
                "filename": filename,
                "page": page,
                "text": chunk_str
            })
            i += (chunk_size - overlap)
    return all_chunks

# --- GOOGLE DRIVE LOADER ---
def load_from_google_drive(url):
    temp_dir = tempfile.mkdtemp()
    downloaded_files = []
    try:
        if "folder" in url:
            output = gdown.download_folder(url, output=temp_dir, quiet=True, proxy=None)
            if output:
                for root, dirs, files in os.walk(temp_dir):
                    for file in files:
                        if file.lower().endswith(('.pdf', '.docx', '.txt', '.md')):
                            downloaded_files.append((os.path.join(root, file), file))
        else:
            output = gdown.download(url, output=temp_dir, fuzzy=True, quiet=True)
            if output:
                filename = os.path.basename(output)
                downloaded_files.append((output, filename))
    except Exception as e:
        st.error(f"Google Drive download failed: {e}")
    return downloaded_files

# --- SESSION STATE INITIALIZATION ---
if "processed_chunks" not in st.session_state:
    st.session_state.processed_chunks = []
if "embeddings" not in st.session_state:
    st.session_state.embeddings = None
if "faiss_index" not in st.session_state:
    st.session_state.faiss_index = None

# --- SIDEBAR: CONFIG & INGESTION ---
st.sidebar.title("📁 Document Ingestion")

groq_api_key = None
try:
    groq_api_key = st.secrets["GROQ_API_KEY"]
except Exception:
    pass

if not groq_api_key:
    groq_api_key = st.sidebar.text_input("Enter Groq API Key", type="password")
    if not groq_api_key:
        st.sidebar.warning("⚠️ Please configure GROQ_API_KEY in `.streamlit/secrets.toml` or sidebar.")

ingestion_source = st.sidebar.radio("Choose Source", ["Local Upload", "Google Drive Link"])

if ingestion_source == "Local Upload":
    uploaded_files = st.sidebar.file_uploader(
        "Upload PDF, DOCX, TXT, MD", 
        type=["pdf", "docx", "txt", "md"], 
        accept_multiple_files=True
    )
    if uploaded_files and st.sidebar.button("Process Local Files"):
        with st.spinner("Extracting & chunking documents..."):
            new_chunks = []
            for uploaded_file in uploaded_files:
                with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{uploaded_file.name}") as tmp:
                    tmp.write(uploaded_file.getvalue())
                    tmp_path = tmp.name
                docs = extract_document(tmp_path, uploaded_file.name)
                chunks = chunk_text(docs)
                new_chunks.extend(chunks)
                try:
                    os.remove(tmp_path)
                except:
                    pass
            
            if new_chunks:
                st.session_state.processed_chunks.extend(new_chunks)
                texts = [c["text"] for c in st.session_state.processed_chunks]
                embeddings = embedding_model.encode(texts, show_progress_bar=False)
                st.session_state.embeddings = np.array(embeddings).astype("float32")
                
                dim = st.session_state.embeddings.shape[1]
                index = faiss.IndexFlatL2(dim)
                index.add(st.session_state.embeddings)
                st.session_state.faiss_index = index
                st.sidebar.success(f"Successfully processed {len(new_chunks)} chunks!")

elif ingestion_source == "Google Drive Link":
    drive_url = st.sidebar.text_input("Paste Google Drive File/Folder Link")
    if drive_url and st.sidebar.button("Download & Process Drive Link"):
        with st.spinner("Downloading & processing from Google Drive..."):
            files = load_from_google_drive(drive_url)
            new_chunks = []
            for path, filename in files:
                docs = extract_document(path, filename)
                chunks = chunk_text(docs)
                new_chunks.extend(chunks)
            
            if new_chunks:
                st.session_state.processed_chunks.extend(new_chunks)
                texts = [c["text"] for c in st.session_state.processed_chunks]
                embeddings = embedding_model.encode(texts, show_progress_bar=False)
                st.session_state.embeddings = np.array(embeddings).astype("float32")
                
                dim = st.session_state.embeddings.shape[1]
                index = faiss.IndexFlatL2(dim)
                index.add(st.session_state.embeddings)
                st.session_state.faiss_index = index
                st.sidebar.success(f"Processed {len(new_chunks)} chunks from Drive!")
            else:
                st.sidebar.warning("No supported files found or download failed.")

# --- MAIN APP LAYOUT ---
st.title("🤖 RAG Knowledge Assistant")
st.markdown("Upload documents or load them from Google Drive, then ask questions based strictly on their contents.")

total_chunks = len(st.session_state.processed_chunks)
st.metric("Total Active Document Chunks", total_chunks)

if total_chunks > 0:
    with st.expander("🔍 View Processed Documents Summary"):
        filenames = set(c["filename"] for c in st.session_state.processed_chunks)
        st.write(f"**Loaded Files ({len(filenames)}):** {', '.join(filenames)}")
        st.write(f"**Total Chunks:** {total_chunks}")

# --- HYBRID SEARCH FUNCTION ---
def hybrid_search(query, top_k=3):
    if not st.session_state.processed_chunks or st.session_state.faiss_index is None:
        return []
    
    query_vector = embedding_model.encode([query]).astype("float32")
    distances, indices = st.session_state.faiss_index.search(query_vector, min(top_k * 2, len(st.session_state.processed_chunks)))
    
    semantic_scores = {}
    for rank, idx in enumerate(indices[0]):
        if idx < len(st.session_state.processed_chunks):
            score = 1.0 / (1.0 + float(distances[0][rank]))
            semantic_scores[idx] = score

    query_words = set(query.lower().split())
    keyword_scores = {}
    for idx, chunk in enumerate(st.session_state.processed_chunks):
        chunk_words = chunk["text"].lower().split()
        matches = sum(1 for w in query_words if w in chunk_words)
        keyword_scores[idx] = matches / (len(query_words) + 1e-5)

    combined_scores = []
    for idx in range(len(st.session_state.processed_chunks)):
        sem_score = semantic_scores.get(idx, 0.0)
        kw_score = keyword_scores.get(idx, 0.0)
        final_score = (0.7 * sem_score) + (0.3 * kw_score)
        combined_scores.append((idx, final_score))
        
    combined_scores.sort(key=lambda x: x[1], reverse=True)
    top_indices = [idx for idx, score in combined_scores[:top_k]]
    
    return [st.session_state.processed_chunks[idx] for idx in top_indices]

# --- Q&A INTERFACE ---
st.divider()
st.subheader("💬 Ask Questions")

user_question = st.text_input("Enter your question about the documents:")

if st.button("Ask Assistant"):
    if not user_question.strip():
        st.warning("Please enter a question.")
    elif total_chunks == 0:
        st.warning("Please ingest documents first.")
    elif not groq_api_key:
        st.error("Groq API key is missing. Please configure it.")
    else:
        with st.spinner("Searching knowledge base & generating answer..."):
            retrieved = hybrid_search(user_question, top_k=3)
            
            context_str = ""
            for i, chunk in enumerate(retrieved):
                context_str += f"[Source {i+1} - File: {chunk['filename']}, Page: {chunk['page']}]\n{chunk['text']}\n\n"
            
            prompt = f"""You are a precise assistant. Answer the user's question using ONLY the provided context below. 
If the answer cannot be found in the context, state clearly: "Not available in the provided documents."

Context:
{context_str}

User Question: {user_question}
Answer:"""

            try:
                client = Groq(api_key=groq_api_key)
                response = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[
                        {"role": "system", "content": "You answer strictly based on the provided context."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1
                )
                answer = response.choices[0].message.content
                
                st.markdown("### Answer")
                st.markdown(answer)
                
                st.markdown("### Retrieved Sources")
                for i, chunk in enumerate(retrieved):
                    st.markdown(f"""
                    <div class="source-box">
                        <b>Source {i+1}</b><br>
                        <b>Filename:</b> {chunk['filename']} | <b>Page:</b> {chunk['page']}<br>
                        <b>Text Snippet:</b> {chunk['text'][:300]}...
                    </div>
                    """, unsafe_allow_html=True)
                    
            except Exception as e:
                st.error(f"Error calling Groq API: {e}")