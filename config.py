# MODEL_PATH = "models/qwen2-1_5b-instruct-q4_k_m.gguf"
MODEL_PATH = "models/DeepSeek-R1-Distill-Qwen-1.5B-Q6_K_2.gguf"
DOCUMENTS_PATH = "data/documents"


EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


CHUNK_SIZE = 500
CHUNK_OVERLAP = 100


# MAX_TOKENS = 300
TOP_K = 4
RETRIEVAL_FETCH_K = 6
MAX_TOKENS = 300


FAISS_INDEX_PATH = "data/faiss.index"
CHUNKS_PATH = "data/chunks.pkl"
METADATA_PATH = "data/metadata.pkl"
NOTES_PATH = "data/notes"

IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".bmp", ".tiff"]

# OCR routing
HANDWRITTEN_KEYWORDS = ["hand", "note", "written", "hw"]

# TrOCR handwritten checkpoint
TROCR_MODEL_NAME = "microsoft/trocr-small-handwritten"