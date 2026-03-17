import re


def deduplicate_chunks(chunks: list[str]) -> list[str]:
    unique = []
    seen = set()

    for chunk in chunks:
        normalized = " ".join(chunk.split()).strip().lower()
        if normalized and normalized not in seen:
            seen.add(normalized)
            unique.append(chunk)

    return unique


def split_large_paragraph(paragraph: str, chunk_size: int, overlap: int) -> list[str]:
    chunks = []
    start = 0
    step = chunk_size - overlap

    while start < len(paragraph):
        end = min(start + chunk_size, len(paragraph))

        if end < len(paragraph):
            window = paragraph[start:end]
            last_break = max(
                window.rfind(". "),
                window.rfind("! "),
                window.rfind("? "),
                window.rfind("\n")
            )
            if last_break > chunk_size // 2:
                end = start + last_break + 1

        chunk = paragraph[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= len(paragraph):
            break

        start += step

    return chunks


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> list[str]:
    text = text.strip()
    if not text:
        return []

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

    chunks = []
    current_chunk = ""

    for para in paragraphs:
        if len(para) > chunk_size:
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""

            chunks.extend(split_large_paragraph(para, chunk_size, overlap))
            continue

        candidate = f"{current_chunk}\n\n{para}".strip() if current_chunk else para

        if len(candidate) <= chunk_size:
            current_chunk = candidate
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = para

    if current_chunk:
        chunks.append(current_chunk.strip())

    return deduplicate_chunks(chunks)