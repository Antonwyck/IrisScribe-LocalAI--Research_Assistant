import os
import streamlit as st
import time
import hashlib
import numpy as np
from config import (
    MODEL_PATH,
    DOCUMENTS_PATH,
    NOTES_PATH,
    EMBED_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    TOP_K,
    MAX_TOKENS,
    RETRIEVAL_FETCH_K,
    FAISS_INDEX_PATH,
    CHUNKS_PATH,
    METADATA_PATH,
    IMAGE_EXTENSIONS,
    HANDWRITTEN_KEYWORDS,
)


from ingestion.pdf_reader import extract_pdf_text
from ingestion.ocr_reader import extract_text_from_image
from ingestion.trocr_reader import extract_handwritten_text_trocr
from ingestion.chunker import chunk_text
from embeddings.embedder import Embedder
from vector_store.faiss_db import FaissStore
from llm.llm_engine import LLMEngine

@st.cache_resource
def load_embedder():
    return Embedder(EMBED_MODEL)

@st.cache_resource
def load_llm():
    return LLMEngine(MODEL_PATH)

st.set_page_config(page_title="Iris Assistant", layout="wide")
st.markdown("""
<style>

.block-container {
    padding-top: 1rem;
    padding-bottom: 0.7rem;
}

</style>
""", unsafe_allow_html=True)
def ensure_dirs():
    os.makedirs(DOCUMENTS_PATH, exist_ok=True)
    os.makedirs("data", exist_ok=True)
    os.makedirs("data/debug_ocr", exist_ok=True)
    os.makedirs(NOTES_PATH, exist_ok=True)

ensure_dirs()

# --- Session state initialization ---
if "embedder" not in st.session_state:
    st.session_state.embedder = load_embedder()

if "llm" not in st.session_state:
    st.session_state.llm = load_llm()

if "store" not in st.session_state:
    st.session_state.store = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ---------- Helpers ----------


def deduplicate_results(results: list[dict]) -> list[dict]:
    unique_results = []
    seen = set()

    for item in results:
        chunk_text = item["chunk"].strip()
        source = item["metadata"].get("source", "")
        chunk_id = item["metadata"].get("chunk_id", "")

        key = (source, chunk_id, chunk_text)
        if key not in seen:
            seen.add(key)
            unique_results.append(item)

    return unique_results


def limit_chunks_per_source(results: list[dict], max_per_source: int = 1) -> list[dict]:
    limited = []
    counts = {}

    for item in results:
        source = item["metadata"].get("source", "unknown")
        counts.setdefault(source, 0)

        if counts[source] < max_per_source:
            limited.append(item)
            counts[source] += 1

    return limited

def clean_final_answer(text: str) -> str:
    text = text.strip()

    cut_phrases = [
        "Would you like me to",
        "Yes or No?",
        "Thank you for your interest",
        "If there's anything else I can assist you with",
        "Do not guess missing facts",
        "Do not add general commentary",
        "If the answer is not clearly stated in the context",
        "Use only one short paragraph",
        "Prefer exact wording from the context when possible",
    ]

    for phrase in cut_phrases:
        idx = text.find(phrase)
        if idx != -1:
            text = text[:idx].strip()

    if text and text[-1] not in ".!?":
        last_punct = max(text.rfind("."), text.rfind("!"), text.rfind("?"))
        if last_punct != -1:
            text = text[:last_punct + 1]

    return text

def looks_handwritten(filename: str) -> bool:
    name = filename.lower()
    return any(k in name for k in HANDWRITTEN_KEYWORDS)

def clear_saved_index():
    for path in [FAISS_INDEX_PATH, CHUNKS_PATH, METADATA_PATH]:
        if os.path.exists(path):
            os.remove(path)

def build_prompt(retrieved_results, question, chat_history=None):
    context_parts = []

    for item in retrieved_results:
        source = item["metadata"].get("source", "unknown")
        doc_type = item["metadata"].get("type", "unknown")
        chunk = item["chunk"]
        context_parts.append(f"Source: {source} | Type: {doc_type}\nText:\n{chunk}")

    context = "\n\n".join(context_parts)#Do not use outside knowledge.
    history_text = ""
    if chat_history:
        history_text = format_chat_history(chat_history)
#Do not add general commentary, lessons, or moral conclusions.., Answer strictly and only from the provided context.Do not use prior knowledge.
    return f"""
You are Iris Assistant.
- Write one short paragraph (2–4 sentences maximum).
- Do not explain how you derived the answer.
- Do not mention the context or conversation history.
- Do not repeat sentences.
- If the answer is not clearly present in the context, output exactly:
The retrieved context does not clearly answer this.

Context:
{context}

Question: {question}

Final answer:
""".strip()

def get_indexed_files(store: FaissStore):
    indexed = []
    seen = set()

    for meta in store.metadata:
        source = meta.get("source", "unknown")
        source_type = meta.get("type", "unknown")
        key = (source, source_type)

        if key not in seen:
            indexed.append({
                "source": source,
                "type": source_type
            })
            seen.add(key)

    return indexed

def format_source_label(meta: dict) -> str:
    return f"{meta.get('source', 'unknown')} ({meta.get('type', 'unknown')})"

#def build_knowledge_base(embedder: Embedder) -> FaissStore:
    all_chunks = []
    all_metadata = []

    files = os.listdir(DOCUMENTS_PATH)
    pdf_files = [f for f in files if f.lower().endswith(".pdf")]
    image_files = [f for f in files if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS]

    for pdf_file in pdf_files:
        pdf_path = os.path.join(DOCUMENTS_PATH, pdf_file)
        text = extract_pdf_text(pdf_path)

        if not text.strip():
            continue

        chunks = chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)

        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_metadata.append({
                "source": pdf_file,
                "type": "pdf_text",
                "chunk_id": i,
            })

    for image_file in image_files:
        image_path = os.path.join(DOCUMENTS_PATH, image_file)

        try:
            if looks_handwritten(image_file):
                text = extract_handwritten_text_trocr(image_path)
                source_type = "handwritten_trocr"
            else:
                text = extract_text_from_image(image_path, save_debug=True)
                source_type = "printed_tesseract"
        except Exception as e:
            st.warning(f"Skipped {image_file}: {e}")
            continue

        if not text.strip():
            continue

        if len(text) < CHUNK_SIZE:
            chunks = [text]
        else:
            chunks = chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)

        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_metadata.append({
                "source": image_file,
                "type": source_type,
                "chunk_id": i,
            })

    if not all_chunks:
        raise ValueError("No valid text chunks were created from the input files.")

    embeddings = embedder.encode_texts(all_chunks)

    store = FaissStore(dimension=embeddings.shape[1])
    store.add(embeddings, all_chunks, all_metadata)
    store.save(FAISS_INDEX_PATH, CHUNKS_PATH, METADATA_PATH)

    return store

def build_knowledge_base(embedder: Embedder) -> FaissStore:
    all_chunks = []
    all_metadata = []

    all_paths = []

    if os.path.exists(DOCUMENTS_PATH):
        all_paths.extend(
            os.path.join(DOCUMENTS_PATH, f) for f in os.listdir(DOCUMENTS_PATH)
        )

    if os.path.exists(NOTES_PATH):
        all_paths.extend(
            os.path.join(NOTES_PATH, f) for f in os.listdir(NOTES_PATH)
        )

    for file_path in all_paths:
        chunks, metadata = process_single_file(file_path)
        if chunks:
            all_chunks.extend(chunks)
            all_metadata.extend(metadata)

    if not all_chunks:
        raise ValueError("No valid text chunks were created from the input files.")

    embeddings = embedder.encode_texts(all_chunks)

    store = FaissStore(dimension=embeddings.shape[1])
    store.add(embeddings, all_chunks, all_metadata)
    store.save(FAISS_INDEX_PATH, CHUNKS_PATH, METADATA_PATH)

    return store

def load_or_build_store(embedder: Embedder) -> FaissStore:
    if (
        os.path.exists(FAISS_INDEX_PATH)
        and os.path.exists(CHUNKS_PATH)
        and os.path.exists(METADATA_PATH)
    ):
        store = FaissStore(dimension=384)
        store.load(FAISS_INDEX_PATH, CHUNKS_PATH, METADATA_PATH)
        return store

    return build_knowledge_base(embedder)

def get_named_source_chunks(store: FaissStore, question: str):
    question_lower = question.lower()
    matched = []

    for i, chunk in enumerate(store.text_chunks):
        meta = store.metadata[i]
        src = meta.get("source", "").lower()
        if src and src in question_lower:
            matched.append({
                "chunk": chunk,
                "metadata": meta
            })

    return matched

def get_distinct_sources(results: list[dict]) -> list[str]:
    sources = []
    seen = set()

    for item in results:
        source = item["metadata"].get("source", "unknown")
        if source not in seen:
            seen.add(source)
            sources.append(source)

    return sources

def slugify_text(text: str) -> str:
    text = text.strip().lower()
    text = "".join(c if c.isalnum() or c in (" ", "_", "-") else "_" for c in text)
    text = "_".join(text.split())
    return text[:50]


def save_research_note(project: str, title: str, note_text: str) -> str:
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    project_slug = slugify_text(project) if project.strip() else "general"
    title_slug = slugify_text(title) if title.strip() else "note"

    filename = f"{timestamp}_{project_slug}_{title_slug}.txt"
    file_path = os.path.join(NOTES_PATH, filename)

    content = f"""Project: {project}
Title: {title}
Created: {time.strftime("%Y-%m-%d %H:%M:%S")}

Note:
{note_text}
"""

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)

    return file_path

def compute_file_hash(file_path: str) -> str:
    hasher = hashlib.sha256()

    with open(file_path, "rb") as f:
        while True:
            chunk = f.read(8192)
            if not chunk:
                break
            hasher.update(chunk)

    return hasher.hexdigest()

def process_single_file(file_path: str):
    all_chunks = []
    all_metadata = []

    project = ""
    title = ""
    created = ""

    filename = os.path.basename(file_path)
    ext = os.path.splitext(filename)[1].lower()
    file_hash = compute_file_hash(file_path)
    modified_time = os.path.getmtime(file_path)

    if ext == ".pdf":
        text = extract_pdf_text(file_path)
        source_type = "pdf_text"

    elif ext == ".txt":
        project, title, created, text = extract_note_metadata_and_text(file_path)
        source_type = "research_note"

    elif ext in IMAGE_EXTENSIONS:
        if looks_handwritten(filename):
            text = extract_handwritten_text_trocr(file_path)
            source_type = "handwritten_trocr"
        else:
            text = extract_text_from_image(file_path, save_debug=True)
            source_type = "printed_tesseract"
    else:
        return all_chunks, all_metadata

    if not text or not text.strip():
        return all_chunks, all_metadata

    if ext in IMAGE_EXTENSIONS and len(text) < 1000:
        chunks = [text.strip()]
    else:
        chunks = chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)

    for i, chunk in enumerate(chunks):
        all_chunks.append(chunk)
        all_metadata.append({
            "source": filename,
            "type": source_type,
            "chunk_id": i,
            "file_hash": file_hash,
            "modified_time": modified_time,
            "project": project,
            "title": title,
            "created": created,
        })

    return all_chunks, all_metadata

def get_already_indexed_sources(store: FaissStore) -> set[str]:
    sources = set()

    for meta in store.metadata:
        source = meta.get("source")
        if source:
            sources.add(source)

    return sources

def get_indexed_file_hashes(store: FaissStore) -> set[str]:
    hashes = set()

    for meta in store.metadata:
        file_hash = meta.get("file_hash")
        if file_hash:
            hashes.add(file_hash)

    return hashes

def add_files_to_store(store: FaissStore, embedder: Embedder, file_paths: list[str]) -> FaissStore:

    indexed_hashes = get_indexed_file_hashes(store)
    indexed_sources = get_already_indexed_sources(store)

    all_chunks = []
    all_metadata = []

    for file_path in file_paths:

        filename = os.path.basename(file_path)
        file_hash = compute_file_hash(file_path)

        if file_hash in indexed_hashes:
            continue

        if filename in indexed_sources:
            store.remove_source(filename)

        chunks, metadata = process_single_file(file_path)

        if chunks:
            all_chunks.extend(chunks)
            all_metadata.extend(metadata)

    if not all_chunks:
        return store

    embeddings = embedder.encode_texts(all_chunks)

    store.add(embeddings, all_chunks, all_metadata)
    store.save(FAISS_INDEX_PATH, CHUNKS_PATH, METADATA_PATH)

    return store

def format_chat_history(history, max_turns=3):
    recent = history[-max_turns:]
    formatted = []

    for user, assistant in recent:
        formatted.append(f"User: {user}")
        formatted.append(f"Assistant: {assistant}")

    return "\n".join(formatted)

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    vec1 = vec1.reshape(-1)
    vec2 = vec2.reshape(-1)
    return float(np.dot(vec1, vec2))

def rerank_results(query_embedding: np.ndarray, retrieved_results: list[dict], embedder: Embedder) -> list[dict]:
    if not retrieved_results:
        return []

    chunk_texts = [item["chunk"] for item in retrieved_results]
    chunk_embeddings = embedder.encode_texts(chunk_texts)

    latest_time = 0.0
    for item in retrieved_results:
        latest_time = max(latest_time, item["metadata"].get("modified_time", 0.0))

    reranked = []
    for i, item in enumerate(retrieved_results):
        similarity = cosine_similarity(query_embedding, chunk_embeddings[i])

        modified_time = item["metadata"].get("modified_time", 0.0)
        source_type = item["metadata"].get("type", "")

        recency_boost = 0.0
        if latest_time > 0 and modified_time > 0:
            recency_ratio = modified_time / latest_time
            recency_boost = 0.05 * recency_ratio

        note_boost = 0.0
        if source_type == "research_note":
            note_boost = 0.05

        final_score = similarity + recency_boost + note_boost

        new_item = dict(item)
        new_item["rerank_score"] = final_score
        new_item["similarity_score"] = similarity
        new_item["recency_boost"] = recency_boost
        new_item["note_boost"] = note_boost
        reranked.append(new_item)

    reranked.sort(key=lambda x: x["rerank_score"], reverse=True)
    return reranked

def extract_note_metadata_and_text(file_path: str):
    project = ""
    title = ""
    created = ""
    note_lines = []

    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    in_note_section = False

    for line in lines:
        stripped = line.strip()

        if stripped.startswith("Project:"):
            project = stripped.replace("Project:", "", 1).strip()
        elif stripped.startswith("Title:"):
            title = stripped.replace("Title:", "", 1).strip()
        elif stripped.startswith("Created:"):
            created = stripped.replace("Created:", "", 1).strip()
        elif stripped == "Note:":
            in_note_section = True
        elif in_note_section:
            note_lines.append(line.rstrip())

    note_text = "\n".join(note_lines).strip()
    return project, title, created, note_text

def is_retrieval_confident(results: list[dict], max_distance: float = 1.0) -> bool:
    if not results:
        return False

    if "distance" not in results[0]:
        return True

    best_distance = results[0].get("distance", 999.0)
    return best_distance <= max_distance

def get_project_matched_chunks(store: FaissStore, question: str):
    question_lower = question.lower()
    matched = []

    for i, chunk in enumerate(store.text_chunks):
        meta = store.metadata[i]
        project = meta.get("project", "").lower()

        if project and project in question_lower:
            matched.append({
                "chunk": chunk,
                "metadata": meta
            })

    return matched

def is_exact_text_question(question: str) -> bool:
    q = question.lower()
    phrases = [
        "exact text",
        "exact words",
        "what text is written",
        "what is written",
        "full text",
        "ocr text",
        "transcript",
        "provide the text",
    ]
    return any(p in q for p in phrases)

def is_latest_update_question(question: str) -> bool:
    q = question.lower()

    phrases = [
        "latest",
        "recent",
        "recently",
        "newest",
        "last update",
        "most recent",
        "what did i add",
        "latest update",
    ]

    return any(p in q for p in phrases)

def get_latest_project_chunks(store: FaissStore, project_name: str):
    project_name = project_name.lower()
    matches = []

    for i, chunk in enumerate(store.text_chunks):
        meta = store.metadata[i]
        project = meta.get("project", "").lower()

        if project == project_name:
            matches.append({
                "chunk": chunk,
                "metadata": meta
            })

    if not matches:
        return []

    matches.sort(
        key=lambda x: x["metadata"].get("modified_time", 0),
        reverse=True
    )

    return matches

def extract_project_from_question(store: FaissStore, question: str):
    question_lower = question.lower()

    projects = set()

    for meta in store.metadata:
        project = meta.get("project")
        if project:
            projects.add(project.lower())

    for project in projects:
        if project in question_lower:
            return project

    return None

if st.session_state.store is None:
    try:
        with st.spinner("Loading knowledge base..."):
            st.session_state.store = load_or_build_store(st.session_state.embedder)
    except Exception as e:
        st.session_state.store = None
        st.warning(f"Knowledge base not ready yet: {e}")
# ---------- Streamlit App ----------

st.title("Iris Assistant")
st.caption("Offline privacy-preserving document, OCR, and RAG assistant")
if st.session_state.store is not None:
    st.success("Knowledge base loaded.")
    #st.info(f"Indexed chunks: {len(st.session_state.store.text_chunks)}")
else:
    st.warning("Knowledge base not loaded yet.")

# Sidebar
with st.sidebar:
    if st.sidebar.button("Clear Conversation"):
        st.session_state.chat_history = []
    st.header("Documents")

    st.subheader("Save Research Note")

    with st.form("save_note_form", clear_on_submit=True, enter_to_submit=False):
        st.caption(f"Saved at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        note_project = st.text_input("Project name")
        note_title = st.text_input("Note title")
        # note_body = st.text_area("Note text", height=180)
        note_body = st.text_area(    "Note text",    height=220,
                                     placeholder="Write experiment results, ideas, or observations...")

        save_note_clicked = st.form_submit_button("Save Note", use_container_width=True)

    if save_note_clicked:
        if not note_body.strip():
            st.warning("Enter note text first.")
        else:
            try:
                note_path = save_research_note(note_project, note_title, note_body)

                if st.session_state.store is None:
                    st.session_state.store = load_or_build_store(st.session_state.embedder)

                st.session_state.store = add_files_to_store(
                    st.session_state.store,
                    st.session_state.embedder,
                    [note_path]
                )

                st.success("Research note saved and indexed.")
                st.rerun()

            except Exception as e:
                st.error(f"Failed to save note: {e}")

    existing_files = os.listdir(DOCUMENTS_PATH) if os.path.exists(DOCUMENTS_PATH) else []
    if existing_files:
        st.write("Files in data/documents:")
        for f in existing_files:
            st.write(f"- {f}")
    else:
        st.write("No files in data/documents yet.")

    if st.session_state.store is not None:
        st.subheader("Indexed Files")
        indexed_files = get_indexed_files(st.session_state.store)

        if indexed_files:
            for item in indexed_files:
                st.write(f"- {item['source']} ({item['type']})")
        else:
            st.write("No indexed files yet.")

    uploaded_files = st.file_uploader(
        "Upload PDFs or images",
        type=["pdf", "png", "jpg", "jpeg", "bmp", "tiff"],
        accept_multiple_files=True
    )
    
    if uploaded_files and st.button("Save Uploaded Files", use_container_width=True):
        saved_paths = []

        for uploaded_file in uploaded_files:
            save_path = os.path.join(DOCUMENTS_PATH, uploaded_file.name)
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            saved_paths.append(save_path)

        if saved_paths:
            st.success("Uploaded file(s) saved to data/documents")

            try:
                with st.spinner("Updating knowledge base..."):
                    if st.session_state.store is None:
                        st.session_state.store = load_or_build_store(st.session_state.embedder)

                    st.session_state.store = add_files_to_store(
                        st.session_state.store,
                        st.session_state.embedder,
                        saved_paths
                    )

                st.success("Knowledge base updated with new files.")
                st.rerun()
            except Exception as e:
                st.error(f"Upload succeeded, but incremental indexing failed: {e}")

    if st.button("Rebuild Knowledge Base", use_container_width=True):
        try:
            clear_saved_index()
            with st.spinner("Rebuilding knowledge base..."):
                st.session_state.store = build_knowledge_base(st.session_state.embedder)
            st.success("Knowledge base rebuilt.")
            st.rerun()
            st.info(f"Indexed chunks: {len(st.session_state.store.text_chunks)}")
        except Exception as e:
            st.error(f"Failed to rebuild knowledge base: {e}")

    if st.button("Refresh Existing Index", use_container_width=True):
        try:
            with st.spinner("Reloading saved index..."):
                st.session_state.store = load_or_build_store(st.session_state.embedder)
            st.success("Index refreshed.")
        except Exception as e:
            st.error(f"Failed to load index: {e}")

# Main QA area
st.subheader("Ask a Question")
# question = st.text_input("Enter your question")
# ask_clicked = st.button("Ask Iris", use_container_width=True)

with st.form("iris_question", clear_on_submit=False):
    question = st.text_input("Enter your question")
    ask_clicked = st.form_submit_button("Ask Iris", use_container_width=True)

if ask_clicked:
    if not question.strip():
        st.warning("Enter a question first.")
    elif st.session_state.store is None:
        st.error("Knowledge base is not loaded.")
    else:
        store = st.session_state.store
        embedder = st.session_state.embedder
        llm = st.session_state.llm

        total_start = time.time()

        with st.spinner("Retrieving context..."):
            match_start = time.time()
            matched_source_chunks = get_named_source_chunks(store, question)
            matched_project_chunks = get_project_matched_chunks(store, question)
            match_time = time.time() - match_start

            embed_time = 0
            retrieval_time = 0
            rerank_time = 0
            project_name = extract_project_from_question(store, question)
            latest_question = is_latest_update_question(question)

            if matched_source_chunks:
                if is_exact_text_question(question) or len(matched_source_chunks) <= 3:
                    retrieved_results = matched_source_chunks
                else:
                    retrieved_results = matched_source_chunks[:TOP_K]
            elif matched_project_chunks:
                if project_name and latest_question:
                    latest_chunks = get_latest_project_chunks(store, project_name)
                    retrieved_results = deduplicate_results(latest_chunks)
                    retrieved_results = limit_chunks_per_source(retrieved_results, max_per_source=2)
                    retrieved_results = retrieved_results[:TOP_K]
                else:
                    retrieved_results = deduplicate_results(matched_project_chunks)
                    retrieved_results = limit_chunks_per_source(retrieved_results, max_per_source=2)
                    retrieved_results = retrieved_results[:TOP_K]

                retrieval_confident = True

            else:
                embed_start = time.time()
                query_embedding = embedder.encode_query(question)
                embed_time = time.time() - embed_start

                retrieval_start = time.time()
                raw_results = store.search(query_embedding, top_k=RETRIEVAL_FETCH_K)
                retrieval_time = time.time() - retrieval_start

                raw_results = deduplicate_results(raw_results)
                retrieval_confident = is_retrieval_confident(raw_results, max_distance=1.0)

                rerank_start = time.time()
                retrieved_results = rerank_results(query_embedding, raw_results, embedder)
                rerank_time = time.time() - rerank_start

                retrieved_results = retrieved_results[:TOP_K]

            # Clean retrieval results in both paths
            retrieved_results = deduplicate_results(retrieved_results)
            #retrieved_results = limit_chunks_per_source(retrieved_results, max_per_source=2)

            distinct_sources = get_distinct_sources(retrieved_results)
            retrieval_confident = is_retrieval_confident(retrieved_results, max_distance=1.0)

        if not retrieved_results:
            st.error("No relevant context found.")
        else:
            # Debug info
            st.write(f"Retrieval confident: {retrieval_confident}")
            st.write(f"Rerank time: {rerank_time:.3f} sec")
            st.write("Top retrieved source:", retrieved_results[0]["metadata"].get("source", "unknown"))

            question_lower = question.lower()
            source_mentioned = any(src.lower() in question_lower for src in distinct_sources)

            if len(distinct_sources) > 1 and not source_mentioned:
                st.warning(
                    "Multiple sources may match this question. Include the filename for a more precise answer."
                )

            if not retrieval_confident and not is_exact_text_question(question):
                final_answer = "The retrieved context does not clearly answer this."
                prompt_time = 0
                generation_time = 0
                ui_update_time = 0.0
                token_count = 0

                total_time = time.time() - total_start

                st.subheader("Answer")
                st.write(final_answer)

                st.subheader("Sources used")
                used_sources = []
                for item in retrieved_results:
                    used_sources.append(format_source_label(item["metadata"]))

                for s in dict.fromkeys(used_sources):
                    st.write(f"- {s}")

            else:
                ocr_text = "\n\n".join([item["chunk"] for item in retrieved_results])

                prompt_time = 0
                generation_time = 0
                ui_update_time = 0.0
                token_count = 0
                final_answer = ""

                if is_exact_text_question(question):
                    final_answer = ocr_text
                else:
                    prompt_start = time.time()
                    prompt = build_prompt(
                        retrieved_results,
                        question,
                        st.session_state.chat_history
                    )
                    prompt_time = time.time() - prompt_start

                    with st.spinner("Generating answer..."):
                        response_placeholder = st.empty()
                        full_answer = ""

                        model_start = time.time()
                        last_ui_update = time.time()

                        for token in llm.generate_stream(prompt, max_tokens=MAX_TOKENS):
                            full_answer += token
                            token_count += 1

                            now = time.time()
                            if now - last_ui_update >= 0.25:
                                ui_start = time.time()
                                response_placeholder.markdown(full_answer)
                                ui_update_time += time.time() - ui_start
                                last_ui_update = now

                        ui_start = time.time()
                        response_placeholder.markdown(full_answer)
                        ui_update_time += time.time() - ui_start

                        generation_time = time.time() - model_start
                        final_answer = clean_final_answer(full_answer)

                        response_placeholder.empty()

                # Save only grounded/valid answers to memory
                st.session_state.chat_history.append((question, final_answer))

                total_time = time.time() - total_start

                st.subheader("Answer")
                st.write(final_answer)

                st.subheader("Sources used")
                used_sources = []
                for item in retrieved_results:
                    used_sources.append(format_source_label(item["metadata"]))

                for s in dict.fromkeys(used_sources):
                    st.write(f"- {s}")

            with st.expander("Retrieved Context", expanded=False):
                for idx, item in enumerate(retrieved_results, start=1):
                    meta = item["metadata"]

                    st.markdown(
                        f"**Chunk {idx}**  \n"
                        f"**Source:** `{meta.get('source', 'unknown')}`  \n"
                        f"**Type:** `{meta.get('type', 'unknown')}`  \n"
                        f"**Chunk ID:** `{meta.get('chunk_id', 'unknown')}`"
                    )
                    st.code(item["chunk"])

            with st.expander("Performance Debug", expanded=False):
                st.write(f"Source match time: {match_time:.3f} sec")
                st.write(f"Query embed time: {embed_time:.3f} sec")
                st.write(f"Retrieval time: {retrieval_time:.3f} sec")
                st.write(f"Rerank time: {rerank_time:.3f} sec")
                st.write(f"Prompt build time: {prompt_time:.3f} sec")
                st.write(f"Generation time: {generation_time:.3f} sec")
                st.write(f"Total time: {total_time:.3f} sec")
                st.write(f"UI update time: {ui_update_time:.3f} sec")
                st.write(f"Token count: {token_count}")
                st.write(f"TOP_K: {TOP_K}, MAX_TOKENS: {MAX_TOKENS}")
                st.write(f"Distinct sources: {distinct_sources}")
                st.write(f"Retrieval confident: {retrieval_confident}")
                if retrieved_results:
                    st.write(f"Best retrieval distance: {retrieved_results[0].get('distance', 'N/A')}")
           