import os
import cv2
import numpy as np
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
os.makedirs("data/debug_ocr", exist_ok=True)
processor = None
model = None


def get_trocr_model():
    global processor, model
    if processor is None or model is None:
        print("IRIS: loading TrOCR model...")
        processor = TrOCRProcessor.from_pretrained("microsoft/trocr-small-handwritten")
        model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-small-handwritten")
    return processor, model


def preprocess_handwriting(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    thresh = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        15
    )

    return gray, thresh


def segment_lines_by_rows(gray, thresh, base_name="hand"):
    row_sums = np.sum(thresh > 0, axis=1)
    text_rows = row_sums > max(20, int(0.02 * thresh.shape[1]))

    line_ranges = []
    in_line = False
    start = 0

    for i, val in enumerate(text_rows):
        if val and not in_line:
            start = i
            in_line = True
        elif not val and in_line:
            end = i
            if end - start > 20:
                line_ranges.append((start, end))
            in_line = False

    if in_line:
        end = len(text_rows) - 1
        if end - start > 20:
            line_ranges.append((start, end))

    line_images = []
    debug_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    for idx, (y1, y2) in enumerate(line_ranges):
        pad = 8
        y1p = max(0, y1 - pad)
        y2p = min(gray.shape[0], y2 + pad)

        line_thresh = thresh[y1p:y2p, :]
        col_sums = np.sum(line_thresh > 0, axis=0)
        ink_cols = np.where(col_sums > 0)[0]

        if len(ink_cols) == 0:
            continue

        x1 = max(0, ink_cols[0] - 10)
        x2 = min(gray.shape[1], ink_cols[-1] + 10)

        cropped = gray[y1p:y2p, x1:x2]
        line_images.append(cropped)

        cv2.rectangle(debug_img, (x1, y1p), (x2, y2p), (0, 255, 0), 2)
        cv2.imwrite(f"data/debug_ocr/{base_name}_line_{idx}.png", cropped)

    cv2.imwrite(f"data/debug_ocr/{base_name}_lines_debug.png", debug_img)
    return line_images


def recognize_line(line_img):
    processor, model = get_trocr_model()

    pil_img = Image.fromarray(line_img).convert("RGB")
    pixel_values = processor(images=pil_img, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values, max_new_tokens=64)
    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return text.strip()


def extract_handwritten_text_trocr(image_path,save_debug=False):
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    gray, thresh = preprocess_handwriting(image_path)

    cv2.imwrite(f"data/debug_ocr/{base_name}_gray.png", gray)
    cv2.imwrite(f"data/debug_ocr/{base_name}_thresh.png", thresh)

    line_images = segment_lines_by_rows(gray, thresh, base_name=base_name)

    if not line_images:
        return ""

    lines = []
    for i, line_img in enumerate(line_images):
        text = recognize_line(line_img)
        print(f"IRIS: TrOCR line {i}: {text}")
        if text:
            lines.append(text)

    return "\n".join(lines)