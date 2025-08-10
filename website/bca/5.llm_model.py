import os
import re
import json
import argparse
from typing import List, Dict, Any, Optional

import chromadb
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

CHROMA_HOST = os.getenv("CHROMA_HOST", "110.239.80.161")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "promo_collection")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

_model: Optional[SentenceTransformer] = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer("LazarusNLP/all-indo-e5-small-v4")
    return _model


def base_score(distance: float) -> float:
    return 1.0 - float(distance)


# -------------------------------------------------------------------
# Query Chroma
# -------------------------------------------------------------------
def search_promos(query_text: str, n_results: int = 5) -> List[Dict[str, Any]]:
    client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
    collection = client.get_collection(name=CHROMA_COLLECTION)

    model = get_model()
    q_emb = model.encode(
        [query_text], normalize_embeddings=True, convert_to_numpy=True
    ).tolist()

    raw = collection.query(
        query_embeddings=q_emb,
        n_results=max(n_results * 8, 40),
        include=["documents", "metadatas", "distances"],
    )

    items = []
    for i, doc in enumerate(raw["documents"][0]):
        meta = raw["metadatas"][0][i] or {}
        dist = raw["distances"][0][i]
        pid = meta.get("parent_id") or meta.get("id") or f"row-{i}"
        sim = base_score(dist)
        items.append(
            {
                "parent_id": pid,
                "score": sim,
                "base_similarity": sim,
                "document": doc,
                "meta": meta,
            }
        )

    best_by_parent = {}
    for it in items:
        pid = it["parent_id"]
        if pid not in best_by_parent or it["score"] > best_by_parent[pid]["score"]:
            best_by_parent[pid] = it

    reranked = sorted(best_by_parent.values(), key=lambda x: x["score"], reverse=True)[
        :n_results
    ]

    output = []
    for rank, it in enumerate(reranked, start=1):
        m = it["meta"]
        output.append(
            {
                "rank": rank,
                "title": m.get("title", ""),
                "period": m.get("period", ""),
                "url": m.get("url", ""),
                "category": m.get("category", ""),
                "bank": m.get("bank", ""),
                "payment_methods": m.get("payment_methods", ""),
                "similarity_percent": round(it["base_similarity"] * 100, 2),
                "score_after_boost": round(
                    it["score"], 4
                ),  # tetap dicetak untuk kompatibilitas
                "description": it["document"],
            }
        )
    return output


# -------------------------------------------------------------------
# LLM CONTEXT
# -------------------------------------------------------------------
def summarize_items(items: List[Dict[str, Any]], max_desc_chars=600):
    out = []
    for it in items:
        desc = (it.get("description") or "").strip().replace("\n", " ")
        if len(desc) > max_desc_chars:
            desc = desc[:max_desc_chars].rsplit(" ", 1)[0] + "…"
        out.append(
            {
                "title": it.get("title"),
                "url": it.get("url"),
                "payment_methods": it.get("payment_methods"),
                "period": it.get("period"),
                "category": it.get("category"),
                "bank": it.get("bank"),
                "description": desc,
            }
        )
    return out


SYSTEM_PROMPT = (
    "You are an expert customer support and promotional information assistant for Indonesian banking promotions. "
    "Always respond in fluent Indonesian, regardless of the user's language. "
    "Use ONLY the provided JSON promo data as your source of truth. "
    "If the answer is uncertain or missing, say so honestly and suggest checking the URL."
)


def build_user_message(context_items: List[Dict[str, Any]], user_question: str) -> str:
    context_json = json.dumps(context_items, ensure_ascii=False, separators=(",", ":"))
    return (
        "Berikut data promo dalam JSON:\n"
        f"{context_json}\n\n"
        "Tugas:\n"
        "1) Jawab dalam bahasa Indonesia yang jelas dan santai.\n"
        "2) Gunakan hanya data di atas (jangan mengarang).\n"
        "3) Jika ada banyak promo, urutkan berdasarkan relevansi dengan pertanyaan.\n"
        "4) Sertakan tanggal periode, benefit utama, metode pembayaran, deskripsi lengkap dan URL.\n\n"
        f"Pertanyaan: {user_question}"
    )


# -------------------------------------------------------------------
# LLM
# -------------------------------------------------------------------
def ask_llm(
    user_question: str,
    top_k: int = 8,
    max_desc_chars: int = 600,
    fallback_json_path: str = "bca.json",
    allow_fallback: bool = True,
) -> str:
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY not found")

    context: List[Dict[str, Any]] = []
    try:
        context = search_promos(user_question, n_results=top_k)
    except Exception as e:
        print(f"[WARN] Gagal query ChromaDB: {e}")

    if not context and allow_fallback:
        try:
            with open(fallback_json_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            context = [
                {
                    "title": it.get("title", ""),
                    "url": it.get("url", ""),
                    "payment_methods": it.get("payment_methods", ""),
                    "period": it.get("period", ""),
                    "category": it.get("category", ""),
                    "bank": it.get("bank", ""),
                    "description": it.get("description", ""),
                }
                for it in raw[:top_k]
            ]
            print(f"[INFO] fallback: {fallback_json_path}")
        except Exception as e:
            raise RuntimeError(
                "No context found and fallback JSON loading failed."
            ) from e

    context_items = summarize_items(context, max_desc_chars=max_desc_chars)
    user_message = build_user_message(context_items, user_question)

    client = Groq(api_key=GROQ_API_KEY)
    chat = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        temperature=0.2,
        max_tokens=2000,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
    )
    return chat.choices[0].message.content


# -------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Promo Assistant (Chroma + LLM) — single file (no month logic)"
    )
    parser.add_argument(
        "--mode",
        choices=["chroma", "llm"],
        default="llm",
        help="Mode eksekusi: 'chroma' untuk lihat hasil pencarian saja, 'llm' untuk jawab via LLM",
    )
    parser.add_argument(
        "--q",
        required=True,
        help="question to ask or search query (e.g. 'Ada promo restaurant ga?')",
    )
    parser.add_argument("--k", type=int, default=8, help="Top K konteks dari ChromaDB")
    parser.add_argument(
        "--max-desc", type=int, default=600, help="Batas karakter ringkasan deskripsi"
    )
    parser.add_argument(
        "--fallback", default="bca.json", help="Path fallback JSON jika Chroma gagal"
    )
    parser.add_argument(
        "--no-fallback", action="store_true", help="Matikan fallback ke file JSON lokal"
    )
    args = parser.parse_args()

    if args.mode == "chroma":
        results = search_promos(args.q, n_results=args.k)
        print(json.dumps(results, ensure_ascii=False, indent=2))
    else:
        answer = ask_llm(
            user_question=args.q,
            top_k=args.k,
            max_desc_chars=args.max_desc,
            fallback_json_path=args.fallback,
            allow_fallback=not args.no_fallback,
        )
        print(answer)


if __name__ == "__main__":
    main()
