from homepage import homepage

# ——— Patch 1: Stop Streamlit watcher hitting torch._classes.__path__ ———
import torch
class _DummyPath:
    def __init__(self):
        self._path = []
    def __getattr__(self, name):
        return []
torch._classes.__path__ = _DummyPath()

# ——— Patch 2: Make SentenceTransformer.to() fall back to to_empty() on meta modules ———
import sentence_transformers as _st
_BaseST = _st.SentenceTransformer
class SentenceTransformer(_BaseST):
    def to(self, *args, **kwargs):
        try:
            return super().to(*args, **kwargs)
        except NotImplementedError:
            return super().to_empty(*args, **kwargs)

# ——— Standard imports ———
import streamlit as st
import streamlit.components.v1 as components
import PyPDF2
import numpy as np
from typing import List, Dict
from langdetect import detect, detect_langs
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai
from gtts import gTTS
import speech_recognition as sr
import tempfile, base64, os
import requests, time
import sqlite3
from datetime import datetime
import pandas as pd
import faiss # Import FAISS

# ——— Configuration ———
GENAI_API_KEY = "AIzaSyA5xtoT9HAjH-wsa7OHFXlBjRRcXwCFBMg"
DID_API_KEY = "c3JlZXlhMjIwNjIwMDJAZ21haWwuY29t:JyH-DjFbnXVAxrS7OESV3" # Replace with your actual D-ID API key
AVATAR_IMAGE_URL = "https://raw.githubusercontent.com/de-id/live-streaming-demo/main/alex_v2_idle_image.png"

# Ensure data directories exist
if not os.path.exists("data"):
    os.makedirs("data")
if not os.path.exists("data/pdfs"):
    os.makedirs("data/pdfs")
if not os.path.exists("data/faiss_indexes"):
    os.makedirs("data/faiss_indexes")

# ——— SQLite DB Setup ———
def init_db():
    conn = sqlite3.connect("interactions.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS interactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            timestamp TEXT,
            language TEXT,
            question TEXT,
            answer TEXT,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            filename TEXT NOT NULL,
            filepath TEXT NOT NULL,
            faiss_index_path TEXT NOT NULL,
            language TEXT, -- Store the detected primary language of the document
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    """)
    conn.commit()
    conn.close()

def add_user(username, password):
    conn = sqlite3.connect("interactions.db")
    cursor = conn.cursor()
    try:
        # NOTE: For production, use a strong hashing library like 'bcrypt' or 'passlib'
        # For this example, a simple hash() is used, which is NOT SECURE for real applications.
        cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hash(password)))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False # Username already exists
    finally:
        conn.close()

def verify_user(username, password):
    conn = sqlite3.connect("interactions.db")
    cursor = conn.cursor()
    # NOTE: For production, use a strong hashing library like 'bcrypt' or 'passlib'
    cursor.execute("SELECT id FROM users WHERE username = ? AND password = ?", (username, hash(password)))
    user = cursor.fetchone()
    conn.close()
    return user[0] if user else None

def save_interaction(user_id: int, language: str, question: str, answer: str):
    conn = sqlite3.connect("interactions.db")
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO interactions (user_id, timestamp, language, question, answer)
        VALUES (?, ?, ?, ?, ?)
    """, (user_id, datetime.now().isoformat(), language, question, answer))
    conn.commit()
    conn.close()

def save_document_metadata(user_id: int, filename: str, filepath: str, faiss_index_path: str, language: str):
    conn = sqlite3.connect("interactions.db")
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO documents (user_id, filename, filepath, faiss_index_path, language)
        VALUES (?, ?, ?, ?, ?)
    """, (user_id, filename, filepath, faiss_index_path, language))
    conn.commit()
    conn.close()

def get_user_documents(user_id: int) -> List[Dict]:
    conn = sqlite3.connect("interactions.db")
    cursor = conn.cursor()
    cursor.execute("SELECT id, filename, filepath, faiss_index_path, language FROM documents WHERE user_id = ?", (user_id,))
    docs = [{"id": row[0], "filename": row[1], "filepath": row[2], "faiss_index_path": row[3], "language": row[4]} for row in cursor.fetchall()]
    conn.close()
    return docs

# ——— RAGSingleLanguage class ———
class RAGSingleLanguage:
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        self.embedder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        self.chunks: List[str] = []
        self.faiss_index = None
        self.language: str = 'en' # Default language for translation if not explicitly set

    def detect_languages(self, text: str) -> List[str]:
        seg_size = 1000
        probs = {}
        for i in range(0, len(text), seg_size):
            seg = text[i:i+seg_size]
            try:
                for lang in detect_langs(seg):
                    probs[lang.lang] = max(probs.get(lang.lang, 0.0), lang.prob)
            except:
                continue
        # Only return languages with a probability >= 0.2
        langs = [l for l,p in probs.items() if p >= 0.2]
        # Fallback to English if no strong detection
        return langs or ['en']

    def translate(self, text: str, tgt: str) -> str:
        try:
            src = detect(text)
        except:
            src = 'en' # Assume English if detection fails
        if src.lower() == tgt.lower():
            return text
        prompt = f"Translate to {tgt.upper()}:\n\n{text}"
        try:
            return self.model.generate_content(prompt).text.strip()
        except Exception as e:
            st.warning(f"Translation failed: {e}. Returning original text.")
            return text

    def process_document(self, pdf_file_path: str, chunk_size: int = 500) -> str:
        reader = PyPDF2.PdfReader(pdf_file_path)
        pages = [p.extract_text() or "" for p in reader.pages]
        full_text = " ".join(pages)
        
        # Detect dominant language of the document
        detected_langs = self.detect_languages(full_text)
        # We'll store the first detected language as the document's primary language
        doc_language = detected_langs[0] if detected_langs else 'en'
        
        full = full_text.split()
        self.chunks = [
            " ".join(full[i:i+chunk_size])
            for i in range(0, len(full), chunk_size)
        ]
        
        # Generate embeddings
        embeddings = self.embedder.encode(
            self.chunks,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        # Create FAISS index
        dimension = embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatL2(dimension)
        self.faiss_index.add(embeddings)
        
        return doc_language # Return the detected language for saving

    def load_faiss_index(self, faiss_index_path: str, document_chunks: List[str]):
        try:
            self.faiss_index = faiss.read_index(faiss_index_path)
            self.chunks = document_chunks # Load associated chunks
            return True
        except Exception as e:
            st.error(f"Error loading FAISS index: {e}")
            return False

    def set_language(self, lang: str):
        self.language = lang

    def answer_question(self, question: str, top_k: int = 5) -> str:
        if self.faiss_index is None or not self.chunks:
            return "Please select a document to query from."

        q_en = self.translate(question, 'en')
        q_emb = self.embedder.encode([q_en], convert_to_numpy=True, normalize_embeddings=True)

        # Search FAISS index
        # D, I are distances and indices respectively.
        # For normalized embeddings, L2 distance (d) is related to cosine similarity (s) by d^2 = 2(1-s)
        distances, indices = self.faiss_index.search(q_emb, top_k)
        
        contexts = []
        for i, dist in zip(indices[0], distances[0]):
            if i >= 0 and i < len(self.chunks): # Ensure index is valid
                sim_score = 1 - (dist / 2) # Convert L2 distance to cosine similarity for display
                contexts.append(f"[Score: {sim_score:.2f}]\n{self.chunks[i]}")

        ctx = "\n\n".join(contexts)

        prompt = (
            "Answer the following question using only the provided context. "
            "Be accurate and detailed. If the answer is not present, say: "
            "'I apologize, but I cannot find this information in the documentation. "
            "Please contact customer support for accurate assistance on this matter.'\n\n"
            f"Context:\n{ctx}\n\nQuestion: {q_en}"
        )

        try:
            out = self.model.generate_content(prompt).text.strip()
        except Exception as e:
            return f"Error generating answer: {e}"
        return self.translate(out, self.language)

# ——— Voice Input ———
def recognize_voice(lang_code='en-IN') -> str:
    r = sr.Recognizer()
    with sr.Microphone() as src:
        st.info("🎤 Adjusting for ambient noise…")
        r.adjust_for_ambient_noise(src, duration=1)
        st.info("Listening…")
        try:
            audio = r.listen(src, timeout=10, phrase_time_limit=10)
        except sr.WaitTimeoutError:
            st.warning("⏰ No speech detected.")
            return ""
    try:
        return r.recognize_google(audio, language=lang_code)
    except sr.UnknownValueError:
        st.error("❗ Could not understand audio.")
    except sr.RequestError as e:
        st.error(f"🚫 Speech API error: {e}")
    return ""

# ——— D-ID Avatar Generator ———
def generate_did_avatar_video(answer_text: str, image_url: str) -> str:
    url = "https://api.d-id.com/talks"
    
    headers = {
        "Authorization": f"Basic {base64.b64encode(DID_API_KEY.encode()).decode()}",
        "Content-Type": "application/json"
    }

    payload = {
        "source_url": image_url,
        "script": {
            "type": "text",
            "input": answer_text,
            "provider": {
                "type": "microsoft",
                "voice_id": "en-US-GuyNeural",
                "voice_config": {
                    "style": "Cheerful"
                }
            }
        },
        "config": {
            "stitch": True
        }
    }

    response = requests.post(url, json=payload, headers=headers)
    if response.status_code not in [200, 201]:
        st.error(f"❌ Avatar video request failed: {response.text}")
        return ""

    talk_id = response.json().get("id")
    if not talk_id:
        st.error("❌ Talk ID not found in response.")
        return ""

    # Poll for video status
    for _ in range(30):
        time.sleep(2)
        check = requests.get(f"https://api.d-id.com/talks/{talk_id}", headers=headers)
        if check.status_code == 200:
            data = check.json()
            if data.get("status") == "done":
                return data.get("result_url")
            elif data.get("status") == "error":
                st.error(f"❌ D-ID video generation error: {data.get('error')}")
                return ""
    st.warning("⚠️ Avatar video is still processing or timed out.")
    return ""

    talk_id = response.json().get("id")
    if not talk_id:
        st.error("❌ Talk ID not found in response.")
        return ""
    
    # Poll for video status
    for _ in range(30): # Try for up to 60 seconds (30 * 2 seconds)
        time.sleep(2)
        check = requests.get(f"https://api.d-id.com/talks/{talk_id}", headers=headers)
        if check.status_code == 200:
            data = check.json()
            if data.get("status") == "done":
                return data.get("result_url")
            elif data.get("status") == "error":
                st.error(f"❌ D-ID video generation error: {data.get('error')}")
                return ""
    st.warning("⚠️ Avatar video is still processing or timed out.")
    return ""

# ——— Main App ———
def main():
    init_db()
    st.set_page_config(page_title="SpeakDoc AI: Voice-Enabled PDF Assistant with Avatar Responses", page_icon="🗣️")
    st.title("🗣️ SpeakDoc AI: Voice-Enabled PDF Assistant with Avatar Responses")


    # Initialize all session state variables at the top
    if 'rag' not in st.session_state:
        st.session_state.rag = RAGSingleLanguage(GENAI_API_KEY)
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
        st.session_state.user_id = None
        st.session_state.username = None
    if 'selected_doc_id' not in st.session_state:
        st.session_state.selected_doc_id = None
        st.session_state.selected_doc_chunks = []
    if 'current_doc_language' not in st.session_state: # Stores the language of the currently loaded document
        st.session_state.current_doc_language = 'en'
    if 'interaction_language' not in st.session_state: # Stores the language chosen for interaction (can differ from doc lang)
        st.session_state.interaction_language = 'en'
    if 'voice_q' not in st.session_state: # THIS IS THE FIX FOR THE ATTRIBUTEERROR
        st.session_state.voice_q = ""

    st.sidebar.header("How to use")
    st.sidebar.markdown("""
    1. Login or Sign Up.
    2. Upload PDF(s) to your account.
    3. Select a document from your uploads.
    4. Confirm or change the interaction language.
    5. Type or speak your question.
    6. Read or listen to the AI's answer.
    """)

    if not st.session_state.logged_in:
        st.subheader("User Authentication")
        auth_option = st.radio("Choose an option:", ("Login", "Sign Up"))

        with st.form("auth_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Submit")

            if submitted:
                if auth_option == "Login":
                    user_id = verify_user(username, password)
                    if user_id:
                        st.session_state.logged_in = True
                        st.session_state.user_id = user_id
                        st.session_state.username = username
                        st.success(f"Welcome, {username}!")
                        st.rerun() # Rerun to switch to the main app view
                    else:
                        st.error("Invalid username or password.")
                elif auth_option == "Sign Up":
                    if add_user(username, password):
                        st.success("Account created successfully! Please log in.")
                    else:
                        st.error("Username already exists. Please choose a different one.")
    else:
        st.sidebar.write(f"Logged in as: **{st.session_state.username}**")
        if st.sidebar.button("Logout"):
            st.session_state.logged_in = False
            st.session_state.user_id = None
            st.session_state.username = None
            st.session_state.selected_doc_id = None
            st.session_state.selected_doc_chunks = []
            st.session_state.current_doc_language = 'en'
            st.session_state.interaction_language = 'en'
            st.session_state.voice_q = "" # Reset voice input
            st.session_state.rag = RAGSingleLanguage(GENAI_API_KEY) # Reset RAG instance
            st.rerun()

        st.subheader("Document Management")
        uploaded_file = st.file_uploader("Upload your PDF manual(s)", type="pdf", accept_multiple_files=True)
        
        if uploaded_file:
            for file in uploaded_file:
                # Check if the file (by name) is already uploaded by this user
                existing_docs = get_user_documents(st.session_state.user_id)
                if file.name in [doc['filename'] for doc in existing_docs]:
                    st.info(f"Document '{file.name}' already uploaded by you.")
                    continue # Skip to the next file if already exists

                with st.spinner(f"Processing {file.name}…"):
                    # Save PDF to disk
                    pdf_path = os.path.join("data", "pdfs", file.name)
                    with open(pdf_path, "wb") as f:
                        f.write(file.getbuffer())

                    # Process document and get its primary language
                    doc_language = st.session_state.rag.process_document(pdf_path)
                    
                    # Save FAISS index
                    faiss_index_filename = f"{os.path.splitext(file.name)[0]}_{st.session_state.user_id}.faiss"
                    faiss_index_path = os.path.join("data", "faiss_indexes", faiss_index_filename)
                    faiss.write_index(st.session_state.rag.faiss_index, faiss_index_path)

                    # Save chunks separately (FAISS only stores embeddings, not the text chunks)
                    chunks_filename = f"{os.path.splitext(file.name)[0]}_{st.session_state.user_id}.chunks"
                    chunks_path = os.path.join("data", "faiss_indexes", chunks_filename)
                    with open(chunks_path, "w", encoding="utf-8") as f:
                        # Use a unique delimiter that is unlikely to appear in the text
                        f.write("\n--CHUNK_DELIMITER--\n".join(st.session_state.rag.chunks)) 

                    # Save document metadata to DB
                    save_document_metadata(st.session_state.user_id, file.name, pdf_path, faiss_index_path, doc_language)
                st.success(f"✅ Document '{file.name}' processed and saved!")
            st.rerun() # Rerun to refresh the document list

        # Display and allow selection of user's uploaded documents
        user_docs = get_user_documents(st.session_state.user_id)
        if user_docs:
            doc_options_display = {doc['filename']: doc for doc in user_docs}
            # Add an empty option for "No document selected"
            selected_filename = st.selectbox(
                "Select a document to query:", 
                [""] + list(doc_options_display.keys()),
                key="doc_selector" # Add a key to avoid potential widget errors
            )

            # Logic to load selected document's FAISS index and chunks
            if selected_filename and selected_filename != "":
                selected_doc_info = doc_options_display[selected_filename]
                
                # Check if this document is already loaded
                if st.session_state.selected_doc_id != selected_doc_info['id']:
                    st.session_state.selected_doc_id = selected_doc_info['id']
                    
                    chunks_filename = f"{os.path.splitext(selected_doc_info['filename'])[0]}_{st.session_state.user_id}.chunks"
                    chunks_path = os.path.join("data", "faiss_indexes", chunks_filename)
                    
                    if os.path.exists(chunks_path):
                        with open(chunks_path, "r", encoding="utf-8") as f:
                            st.session_state.selected_doc_chunks = f.read().split("\n--CHUNK_DELIMITER--\n")
                    else:
                        st.error("Error: Chunks file not found for this document.")
                        st.session_state.selected_doc_chunks = []
                        st.session_state.selected_doc_id = None # Invalidate selection

                    if st.session_state.selected_doc_id and \
                       st.session_state.rag.load_faiss_index(selected_doc_info['faiss_index_path'], st.session_state.selected_doc_chunks):
                        st.success(f"Selected document: '{selected_filename}'")
                        # Set the detected language of the document
                        st.session_state.current_doc_language = selected_doc_info['language']
                        st.session_state.interaction_language = selected_doc_info['language'] # Default interaction language to doc's
                        st.session_state.rag.set_language(st.session_state.interaction_language)
                        st.rerun() # Rerun to update language selector and clear old inputs
                    else:
                        st.error(f"Could not load FAISS index for '{selected_filename}'.")
                        st.session_state.selected_doc_id = None
                        st.session_state.rag.faiss_index = None
                        st.session_state.rag.chunks = []
                        st.session_state.current_doc_language = 'en'
                        st.session_state.interaction_language = 'en'
                
                # If a document is selected and loaded, allow language choice for interaction
                if st.session_state.selected_doc_id:
                    st.markdown("---") # Separator for clarity
                    st.markdown("**Choose Interaction Language**")
                    
                    # You could fetch all detected languages from the processed document if desired
                    # For simplicity, we'll offer a few common ones, plus the detected document language
                    available_langs = sorted(list(set(['en', 'hi', 'fr', 'es', 'de', st.session_state.current_doc_language])))
                    # Remove duplicates and ensure the current_doc_language is an option
                    
                    lang_selection = st.selectbox(
                        "Select the language for your question and the AI's answer:",
                        [lang.upper() for lang in available_langs],
                        index=available_langs.index(st.session_state.interaction_language) if st.session_state.interaction_language in available_langs else 0,
                        key="interaction_lang_selector"
                    )
                    
                    if lang_selection:
                        new_lang = lang_selection.lower()
                        if new_lang != st.session_state.interaction_language:
                            st.session_state.interaction_language = new_lang
                            st.session_state.rag.set_language(new_lang)
                            st.rerun() # Rerun to update the question input field's language
                    
                    st.markdown(f"**Asking in:** `{st.session_state.interaction_language.upper()}`")
                    st.markdown("---") # Separator

                    st.markdown("**Type your question**")
                    typed_question = st.text_input(f"Ask in {st.session_state.interaction_language.upper()}:", value=st.session_state.voice_q, key="typed_question_input")
                    
                    st.markdown("**Or use voice input**")
                    if st.button("🎙️ Speak Your Question", key="speak_button"):
                        # Adjust language code for speech recognition based on interaction language
                        recognizer_lang_code = st.session_state.interaction_language
                        if recognizer_lang_code == "en":
                            recognizer_lang_code = "en-IN" # Default to Indian English for better recognition in some cases
                        elif recognizer_lang_code == "hi":
                            recognizer_lang_code = "hi-IN" # Hindi
                        # Add more specific regional codes if necessary for other languages

                        recd_speech = recognize_voice(recognizer_lang_code)
                        if recd_speech:
                            st.session_state.voice_q = recd_speech
                            st.success(f"🎤 You said: {recd_speech}")
                            st.rerun() # Rerun to populate the text input with spoken text
                        else:
                            st.warning("No speech recognized.")

                    # Use the typed input or the voice input if available
                    question_to_process = typed_question or st.session_state.voice_q

                    if st.button("Get Answer", key="get_answer_button") and question_to_process:
                        st.markdown(f"🔍 Question: `{question_to_process}`")
                        with st.spinner("Thinking…"):
                            answer = st.session_state.rag.answer_question(question_to_process)
                        st.markdown(f"**Answer ({st.session_state.interaction_language.upper()}):** {answer}")

                        # Save to DB
                        save_interaction(st.session_state.user_id, st.session_state.interaction_language, question_to_process, answer)

                        # Text-to-Speech (gTTS)
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
                            try:
                                gTTS(text=answer, lang=st.session_state.interaction_language).save(fp.name)
                                mp3_bytes = open(fp.name, "rb").read()
                                b64 = base64.b64encode(mp3_bytes).decode()

                                html = f"""
                                <audio id='player' controls>
                                  <source src='data:audio/mp3;base64,{b64}' type='audio/mp3'/>
                                </audio>
                                <canvas id='canvas' width='300' height='100'></canvas>
                                <script>
                                  const audio = document.getElementById('player');
                                  const canvas = document.getElementById('canvas');
                                  const ctx = canvas.getContext('2d');
                                  const audioCtx = new (window.AudioContext||window.webkitAudioContext)();
                                  const source = audioCtx.createMediaElementSource(audio);
                                  const analyser = audioCtx.createAnalyser();
                                  analyser.fftSize = 256;
                                  source.connect(analyser);
                                  analyser.connect(audioCtx.destination);
                                  const data = new Uint8Array(analyser.frequencyBinCount);
                                  function drawLine() {{
                                    requestAnimationFrame(drawLine);
                                    analyser.getByteTimeDomainData(data);
                                    let sum = 0;
                                    for (let i=0; i<data.length; i++) {{
                                      const v = data[i] - 128;
                                      sum += v*v;
                                    }}
                                    const rms = Math.sqrt(sum/data.length);
                                    const maxLen = canvas.width / 2 * (rms/128);
                                    const y = canvas.height / 2;
                                    ctx.clearRect(0, 0, canvas.width, canvas.height);
                                    ctx.beginPath();
                                    ctx.moveTo((canvas.width / 2) - maxLen, y);
                                    ctx.lineTo((canvas.width / 2) + maxLen, y);
                                    ctx.lineWidth = 4;
                                    ctx.strokeStyle = '#4CAF50';
                                    ctx.stroke();
                                  }}
                                  audio.onplay = () => {{
                                    audioCtx.resume().then(() => drawLine());
                                  }};
                                </script>
                                """
                                components.html(html, height=150)
                            except Exception as e:
                                st.error(f"Error generating audio: {e}. Please ensure gTTS supports '{st.session_state.interaction_language}'.")
                        
                        # Clear voice_q after processing the answer
                        st.session_state.voice_q = "" 

                        st.markdown("### 🧑‍💼 Speaking AI Avatar")
                        with st.spinner("Generating avatar video…"):
                            video_url = generate_did_avatar_video(answer, AVATAR_IMAGE_URL)
                            if video_url:
                                st.video(video_url)
                            else:
                                st.error("Failed to load avatar video.")
                    elif st.button("Get Answer") and not question_to_process:
                        st.warning("Please enter or speak a question.")
            else:
                st.info("Please select a document from your uploaded files to start querying.")
                # Reset RAG if no document is selected
                st.session_state.selected_doc_id = None
                st.session_state.rag.faiss_index = None
                st.session_state.rag.chunks = []
                st.session_state.current_doc_language = 'en'
                st.session_state.interaction_language = 'en'
                st.session_state.rag.set_language('en') # Reset RAG's internal language
        else:
            st.info("No documents uploaded yet. Please upload a PDF to begin.")

    # --- Optional: Admin View ---
    st.sidebar.markdown("---")
    st.sidebar.header("Admin Views")

    if st.sidebar.checkbox("📜 Show Past Interactions"):
        if st.session_state.logged_in:
            conn = sqlite3.connect("interactions.db")
            df = pd.read_sql_query(f"SELECT timestamp, language, question, answer FROM interactions WHERE user_id = {st.session_state.user_id} ORDER BY timestamp DESC", conn)
            if not df.empty:
                st.sidebar.dataframe(df)
            else:
                st.sidebar.info("No past interactions for this user.")
            conn.close()
        else:
            st.sidebar.warning("Please log in to view past interactions.")

    if st.sidebar.checkbox("📂 Show My Uploaded Documents"):
        if st.session_state.logged_in:
            user_docs = get_user_documents(st.session_state.user_id)
            if user_docs:
                df_docs = pd.DataFrame(user_docs)
                st.sidebar.dataframe(df_docs[['filename', 'language']])
            else:
                st.sidebar.info("No documents uploaded yet.")
        else:
            st.sidebar.warning("Please log in to view your uploaded documents.")
    
if __name__ == "__main__":
    if 'start_app' not in st.session_state or not st.session_state.start_app:
        homepage()
    else:
        main()
