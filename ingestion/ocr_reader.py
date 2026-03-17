import os
import cv2
import pytesseract


# Uncomment only if needed
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def preprocess_image(image_path: str):
    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    processed = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        15
    )

    return processed


def save_debug_image(processed_image, original_path: str):
    debug_dir = "data/debug_ocr"
    os.makedirs(debug_dir, exist_ok=True)

    filename = os.path.basename(original_path)
    name, _ = os.path.splitext(filename)
    debug_path = os.path.join(debug_dir, f"{name}_processed.png")

    cv2.imwrite(debug_path, processed_image)
    return debug_path


def extract_text_from_image(image_path: str, save_debug: bool = False) -> str:
    processed = preprocess_image(image_path)

    if save_debug:
        debug_path = save_debug_image(processed, image_path)
        print(f"IRIS: saved preprocessed image to {debug_path}")

    configs = [
        "--oem 3 --psm 6",
        "--oem 3 --psm 11",
        "--oem 3 --psm 4",
    ]

    best_text = ""
    best_cfg = ""

    for cfg in configs:
        text = pytesseract.image_to_string(processed, config=cfg).strip()
        if len(text.replace(" ", "")) > len(best_text.replace(" ", "")):
            best_text = text
            best_cfg = cfg

    if best_cfg:
        print(f"IRIS: OCR selected {best_cfg.split()[-1]}")

    return best_text




# import os
# import cv2
# import pytesseract


# # Uncomment and edit this only if tesseract is not in PATH on Windows:
# # pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


# def preprocess_image(image_path: str, save_debug: bool = True):
#     image = cv2.imread(image_path)

#     if image is None:
#         raise FileNotFoundError(f"Could not read image: {image_path}")

#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#     # Upscale to help OCR read smaller handwriting
#     gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

#     # Light denoising
#     blurred = cv2.GaussianBlur(gray, (3, 3), 0)

#     # Adaptive threshold helps when lighting is uneven
#     processed = cv2.adaptiveThreshold(
#         blurred,
#         255,
#         cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#         cv2.THRESH_BINARY,
#         31,
#         15
#     )

#     if save_debug:
#         debug_dir = "data/debug"
#         os.makedirs(debug_dir, exist_ok=True)

#         filename = os.path.basename(image_path)
#         name, _ = os.path.splitext(filename)
#         debug_path = os.path.join(debug_dir, f"{name}_processed.png")

#         cv2.imwrite(debug_path, processed)
#         print(f"IRIS: saved preprocessed image to {debug_path}")

#     return processed


# def extract_text_from_image(image_path: str) -> str:
#     processed = preprocess_image(image_path, save_debug=True)

#     # Try a few page segmentation modes and keep the best result by length
#     psm_modes = [6, 11, 4]
#     results = []

#     for psm in psm_modes:
#         config = f"--oem 3 --psm {psm}"
#         text = pytesseract.image_to_string(processed, config=config).strip()
#         results.append((psm, text))

#     best_psm, best_text = max(results, key=lambda x: len(x[1].strip()))

#     print(f"IRIS: OCR selected PSM {best_psm}")

#     return best_text







# import cv2
# import pytesseract


# def preprocess_image(image_path: str):
#     image = cv2.imread(image_path)

#     if image is None:
#         raise FileNotFoundError(f"Could not read image: {image_path}")

#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#     # Upscale image for better OCR
#     gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

#     # Denoise
#     gray = cv2.GaussianBlur(gray, (3, 3), 0)

#     # Adaptive threshold often works better for uneven lighting
#     processed = cv2.adaptiveThreshold(
#         gray,
#         255,
#         cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#         cv2.THRESH_BINARY,
#         31,
#         15
#     )

#     return processed


# def extract_text_from_image(image_path: str) -> str:
#     processed = preprocess_image(image_path)

#     # psm 6 = block of text
#     text = pytesseract.image_to_string(processed, config="--oem 3 --psm 6")
#     return text.strip()




# import cv2
# import pytesseract


# def preprocess_image(image_path: str):
#     image = cv2.imread(image_path)

#     if image is None:
#         raise FileNotFoundError(f"Could not read image: {image_path}")

#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#     # Light denoising
#     gray = cv2.GaussianBlur(gray, (3, 3), 0)

#     # Threshold to improve OCR contrast
#     processed = cv2.threshold(
#         gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
#     )[1]

#     return processed


# def extract_text_from_image(image_path: str) -> str:
#     processed = preprocess_image(image_path)

#     text = pytesseract.image_to_string(processed, config="--oem 3 --psm 6")
#     return text.strip()