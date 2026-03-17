# IrisScribe – Local AI Research Assistant

IrisScribe is an offline AI assistant designed for intelligent document analysis and information retrieval.  
The system processes documents such as PDFs and images, extracts text using OCR and handwritten text recognition, converts the extracted content into embeddings, and uses Retrieval-Augmented Generation (RAG) with a local large language model to answer user queries.

Unlike many AI tools that rely on cloud APIs, IrisScribe operates locally, allowing users to analyze sensitive documents while maintaining privacy.

---

## Features

- Offline document analysis
- OCR for scanned images and documents
- Handwritten text recognition using TrOCR
- Semantic search using vector embeddings
- Retrieval-Augmented Generation (RAG)
- Local LLM inference
- Interactive interface using Streamlit

---

## System Pipeline

The IrisScribe system follows a multi-stage pipeline:

1. Document Ingestion  
   Documents such as PDFs or images are uploaded into the system.

2. Text Extraction  
   - Printed text is extracted using OCR.
   - Handwritten text is extracted using the TrOCR model.

3. Text Chunking  
   Extracted text is divided into smaller chunks to enable efficient semantic search.

4. Embedding Generation  
   Each chunk is converted into a vector embedding using a sentence embedding model.

5. Vector Storage  
   Embeddings are stored in a FAISS vector database for efficient similarity search.

6. Query Processing  
   When a user asks a question:
   - The query is converted into an embedding.
   - Similar document chunks are retrieved.

7. Answer Generation  
   Retrieved chunks are passed to the local language model which generates the final answer.

---

## Project Structure
IrisScribe-LocalAI--Research_Assistant/
│
├── main.py
├── config.py
├── requirements.txt
├── README.md
│
├── ingestion/
│ ├── pdf_reader.py
│ ├── ocr_reader.py
│ ├── trocr_reader.py
│ └── chunker.py
│
├── embeddings/
│ └── embedder.py
│
├── vector_store/
│ └── faiss_db.py
│
├── llm/
│ └── llm_engine.py
│
└── rag_store/
---

## Installation

Clone the repository:
git clone https://github.com/Antonwyck/IrisScribe-LocalAI--Research_Assistant.git

cd IrisScribe-LocalAI--Research_Assistant

Install dependencies:
pip install -r requirements.txt


---

## Running the Application

Launch the Streamlit interface:
streamlit run main.py

The application will open in your browser where you can upload documents and ask questions.

---

## Model Requirements

This project uses a **local language model** and **embedding model**.

These models are not included in the repository due to their large size.

Download the required models and place them in the appropriate directories defined in `config.py`.

Example:
models/
qwen2-1_5b-instruct-q4_k_m.gguf

---

## Technologies Used

- Python
- Streamlit
- FAISS
- Sentence Transformers
- TrOCR
- Llama.cpp / GGUF models
- NumPy

---

## Future Improvements

Possible enhancements for future versions include:

- Voice-based interaction
- Multilingual document support
- Advanced document organization and tagging
- Faster vector retrieval and indexing
- Integration with additional document formats

---

## License

This project is intended for research and educational purposes.
