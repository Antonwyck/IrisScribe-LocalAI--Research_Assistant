from paddleocr import PaddleOCR
from PIL import Image
import os

_paddle_ocr = None


def get_paddle_ocr():
    global _paddle_ocr
    if _paddle_ocr is None:
        _paddle_ocr = PaddleOCR(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False
        )
    return _paddle_ocr


def extract_text_from_image_paddle(image_path: str) -> str:
    ocr = get_paddle_ocr()
    result = ocr.predict(input=image_path)

    lines = []

    for page in result:
        rec_texts = page.get("rec_texts", [])
        for text in rec_texts:
            if text and text.strip():
                lines.append(text.strip())

    return "\n".join(lines)


def extract_text_from_pdf_paddle(pdf_path: str) -> str:
    # PaddleOCR supports PDFs as input in the project direction,
    # but the Python path can vary by version, so the safest baseline
    # for your current system is still pdfplumber for machine-readable PDFs.
    raise NotImplementedError("Use pdfplumber for text PDFs; PaddleOCR can be added for scanned PDFs later.")