import os
import json
import uuid
from typing import List, Dict, Any, Iterable, Tuple

import chromadb
from chromadb import HttpClient
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# =========================
# Konfigurasi
# =========================
CHROMA_HOST = os.getenv("CHROMA_HOST", "110.239.80.161")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "promo_collection")

# Model lokal Qwen3 Embedding 8B
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "Qwen/Qwen3-Embedding-8B")

PROMOS_JSON_PATH = os.getenv("PROMOS_JSON_PATH", "promos.json")

# Chunking
MAX_CHARS_PER_CHUNK = int(os.getenv("MAX_CHARS_PER_CHUNK", "1200"))
MIN_CHARS_PER_CHUNK = int(os.getenv("MIN_CHARS_PER_CHUNK", "300"))
OVERLAP_CHARS = int(os.getenv("OVERLAP_CHARS", "100"))

BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))  # turunkan kalau VRAM pas-pasan


# =========================
# Util teks
# =========================
def clean_text(s: str) -> str:
    return (s or "").replace("\r", "\n").strip()


def smart_chunks(
    s: str,
    max_chars: int = MAX_CHARS_PER_CHUNK,
    overlap: int = OVERLAP_CHARS,
    min_chars: int = MIN_CHARS_PER_CHUNK,
) -> List[str]:
    s = clean_text(s)
    if not s:
        return []
    if len(s) <= max_chars:
        return [s] if len(s) >= min_chars else []

    chunks = []
    start = 0
    while start < len(s):
        end = min(start + max_chars, len(s))
        if end < len(s):
            window = s[start:end]
            split_idx = max(
                window.rfind("\n\n"), window.rfind("\n"), window.rfind(". ")
            )
            if split_idx != -1 and split_idx >= min_chars // 2:
                end = start + split_idx + 1
        chunk = s[start:end].strip()
        if len(chunk) >= min_chars:
            chunks.append(chunk)
        start = max(end - overlap, start + 1)
        if start >= len(s):
            break
    return chunks or [s]


def build_documents_from_promo(
    promo: Dict[str, Any],
) -> List[Tuple[str, str, Dict[str, Any]]]:
    title = clean_text(promo.get("title", ""))
    url = promo.get("url")
    description = clean_text(promo.get("description", ""))
    period = clean_text(promo.get("period", ""))
    payment_methods = promo.get("payment_methods", [])
    category = promo.get("category")
    bank = promo.get("bank")
    idx = promo.get("index")
    scrape_date = promo.get("scrape_date")
    promo_id = promo.get("id") or uuid.uuid4().hex

    base_text = "\n\n".join(
        [
            part
            for part in [
                f"Title: {title}",
                f"Period: {period}" if period else "",
                f"Description:\n{description}" if description else "",
            ]
            if part
        ]
    ).strip()

    chunks = smart_chunks(base_text)
    docs = []
    for i, ch in enumerate(chunks):
        doc_id = f"{promo_id}::chunk-{i}"
        metadata = {
            "promo_id": promo_id,
            "chunk_index": i,
            "title": title,
            "url": url,
            "period": period,
            "category": category,
            "bank": bank,
            "payment_methods": payment_methods,
            "source_index": idx,
            "scrape_date": scrape_date,
        }
        docs.append((doc_id, ch, metadata))

    if not docs:
        doc_id = f"{promo_id}::chunk-0"
        docs = [
            (
                doc_id,
                f"Title: {title}\nURL: {url}\nPeriod: {period}",
                {
                    "promo_id": promo_id,
                    "chunk_index": 0,
                    "title": title,
                    "url": url,
                    "period": period,
                    "category": category,
                    "bank": bank,
                    "payment_methods": payment_methods,
                    "source_index": idx,
                    "scrape_date": scrape_date,
                },
            )
        ]
    return docs


def batched(iterable, n: int):
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= n:
            yield batch
            batch = []
    if batch:
        yield batch


# =========================
# Main
# =========================
def main():
    # 1) Connect Chroma
    print(f"Connecting to Chroma @ http://{CHROMA_HOST}:{CHROMA_PORT}")
    client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)

    # 2) Recreate collection
    try:
        client.delete_collection(COLLECTION_NAME)
        print(f"[Chroma] Deleted existing collection: {COLLECTION_NAME}")
    except Exception as e:
        print(f"[Chroma] Delete skipped ({e})")

    collection = client.create_collection(name=COLLECTION_NAME)
    print(f"[Chroma] Created collection: {COLLECTION_NAME}")

    # 3) Load model lokal Qwen3 8B
    #    Rekomendasi dari model card: flash_attention_2 + padding_side='left'
    print(f"Loading embedding model: {EMBED_MODEL_NAME}")
    model = SentenceTransformer(
        EMBED_MODEL_NAME,
        device="cuda" if os.environ.get("CUDA_VISIBLE_DEVICES", "") != "" else None,
        model_kwargs={"attn_implementation": "flash_attention_2", "device_map": "auto"},
        tokenizer_kwargs={"padding_side": "left"},
    )

    # 4) Load promos
    with open(PROMOS_JSON_PATH, "r", encoding="utf-8") as f:
        promos = json.load(f)
    if isinstance(promos, dict):
        promos = [promos]
    print(f"Loaded {len(promos)} promos from {PROMOS_JSON_PATH}")

    # 5) Build chunks
    all_docs: List[Tuple[str, str, Dict[str, Any]]] = []
    for promo in promos:
        all_docs.extend(build_documents_from_promo(promo))
    print(f"Prepared {len(all_docs)} chunks")

    # 6) Embed & upsert
    pbar = tqdm(total=len(all_docs), desc="Embedding & upserting")
    for batch in batched(all_docs, BATCH_SIZE):
        ids = [doc_id for (doc_id, _, _) in batch]
        texts = [text for (_, text, _) in batch]
        metadatas = [meta for (_, _, meta) in batch]

        # Dokumen TIDAK pakai prompt (sesuai rekomendasi)
        doc_embeddings = model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=False,  # biar hemat mem
            batch_size=BATCH_SIZE,
        )
        # simpan
        collection.add(
            ids=ids,
            documents=texts,
            embeddings=[e.tolist() for e in doc_embeddings],
            metadatas=metadatas,
        )
        pbar.update(len(batch))
    pbar.close()

    print(f"Done. Count in `{COLLECTION_NAME}` = {collection.count()}")


if __name__ == "__main__":
    main()
