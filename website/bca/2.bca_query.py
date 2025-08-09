import chromadb
import cohere
import os
from collections import defaultdict
from dotenv import load_dotenv

load_dotenv()

CHROMA_HOST = "110.239.80.161"
CHROMA_PORT = 8000
COLLECTION_NAME = "promo_collection"

COHERE_API_KEY = os.getenv("cohere_api", "")
COHERE_MODEL = "embed-multilingual-v3.0"


def get_clients():
    chroma = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
    col = chroma.get_or_create_collection(name=COLLECTION_NAME)
    co = cohere.Client(COHERE_API_KEY)
    return col, co


def embed_query(co: cohere.Client, text: str):
    out = co.embed(
        model=COHERE_MODEL,
        texts=[text],
        input_type="search_query",
        truncate="NONE",
    )
    return out.embeddings[0]


def search_promos(query: str, top_k: int = 8, filters: dict = None):
    col, co = get_clients()
    qvec = embed_query(co, query)

    res = col.query(
        query_embeddings=[qvec],
        n_results=top_k,
        where=filters or {},
        include=["documents", "metadatas", "distances"],
    )

    ids = res.get("ids", [[]])[0]
    dists = res.get("distances", [[]])[0]
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]

    grouped = {}
    for i, mid in enumerate(ids):
        m = metas[i]
        parent = m.get("parent_id", mid)
        score = 1.0 - dists[i] if dists else None
        if parent not in grouped:
            grouped[parent] = {
                "title": m.get("title", ""),
                "url": m.get("url", ""),
                "bank": m.get("bank", ""),
                "category": m.get("category", ""),
                "period": m.get("period", ""),
                "payment_methods": m.get("payment_methods", ""),
                "best_score": score,
                "snippets": [docs[i]],
            }
        else:
            grouped[parent]["snippets"].append(docs[i])
            if score is not None and (
                grouped[parent]["best_score"] is None
                or score > grouped[parent]["best_score"]
            ):
                grouped[parent]["best_score"] = score

    results = sorted(
        grouped.values(), key=lambda x: (x["best_score"] or 0), reverse=True
    )
    return results


def print_results(results, limit=5):
    for i, r in enumerate(results[:limit], 1):
        print(f"{i}. {r['title']}  [{r['best_score']:.4f}]")
        print(f"   URL     : {r['url']}")
        if r.get("period"):
            print(f"   Periode : {r['period']}")
        if r.get("payment_methods"):
            print(f"   Pembayaran: {r['payment_methods']}")
        if r["snippets"]:
            snippet = r["snippets"][0].strip().replace("\n", " ")
            print(f"   Snippet : {snippet[:200]}{'...' if len(snippet)>200 else ''}")
        print()


if __name__ == "__main__":
    q = "ada promo telkomsel ga"
    results = search_promos(q, top_k=12, filters={"bank": "BCA"})

    print_results(results, limit=5)
