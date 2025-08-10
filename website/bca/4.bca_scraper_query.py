import json
import re
import chromadb
from sentence_transformers import SentenceTransformer

CHROMA_HOST = "110.239.80.161"
CHROMA_PORT = 8000
COLLECTION_NAME = "promo_collection"

client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
collection = client.get_collection(name=COLLECTION_NAME)

model = SentenceTransformer("LazarusNLP/all-indo-e5-small-v4")

query_text = "Ada promo restaurant ga ?"


def month_table():
    return {
        "01": ["januari", "jan", "jan.", "january", "janv", "01", "1"],
        "02": ["februari", "feb", "feb.", "february", "02", "2"],
        "03": ["maret", "mar", "mar.", "march", "03", "3"],
        "04": ["april", "apr", "apr.", "april.", "04", "4"],
        "05": ["mei", "may", "may.", "mei.", "05", "5"],
        "06": ["juni", "jun", "jun.", "june", "06", "6"],
        "07": ["juli", "jul", "jul.", "july", "07", "7"],
        "08": ["agustus", "agus", "aug", "aug.", "august", "08", "8"],
        "09": ["september", "sep", "sept", "sept.", "09", "9"],
        "10": ["oktober", "okt", "okt.", "oct", "oct.", "october", "10"],
        "11": ["november", "nov", "nov.", "11"],
        "12": ["desember", "des", "des.", "dec", "dec.", "december", "12"],
    }


def detect_months(q):
    ql = q.lower()
    mts = month_table()
    found = set()
    for mm, syns in mts.items():
        for s in syns:
            if re.search(rf"\b{re.escape(s)}\b", ql):
                found.add(mm)
                break
    return sorted(found)


def expand_synonyms(months):
    mts = month_table()
    out = []
    for mm in months:
        out.extend(mts.get(mm, []))
        out.append(mm)
    return sorted(set(out))


months_in_query = detect_months(query_text)
syns = expand_synonyms(months_in_query)
q_aug = "query: " + query_text + (" " + " ".join(syns) if syns else "")
q_emb = model.encode([q_aug], normalize_embeddings=True, convert_to_numpy=True).tolist()

raw = collection.query(
    query_embeddings=q_emb,
    n_results=8,
    include=["documents", "metadatas", "distances"],
)


def base_score(distance):
    return 1.0 - float(distance)


def month_boost(meta, doc, months):
    text = " ".join(
        [str(meta.get("period", "")), str(meta.get("title", "")), doc or ""]
    ).lower()
    mts = month_table()
    b = 0.0
    for mm in months:
        syns = mts.get(mm, [])
        if any(re.search(rf"\b{re.escape(s)}\b", text) for s in syns):
            b += 0.12
        if re.search(r"\b20\d{2}\b", text) and (
            re.search(rf"\b{mm}\b", text)
            or any(re.search(rf"\b0?{int(mm)}\b", text) for _ in [0])
        ):
            b += 0.05
        prevm = f"{(int(mm)-2)%12+1:02d}"
        nextm = f"{int(mm)%12+1:02d}"
        syn_prev = mts.get(prevm, [])
        syn_next = mts.get(nextm, [])
        if (
            any(re.search(rf"\b{re.escape(s)}\b", text) for s in syns)
            and any(re.search(rf"\b{re.escape(s)}\b", text) for s in syn_next)
        ) or (
            any(re.search(rf"\b{re.escape(s)}\b", text) for s in syns)
            and any(re.search(rf"\b{re.escape(s)}\b", text) for s in syn_prev)
        ):
            b += 0.08
    return b


items = []
for i, doc in enumerate(raw["documents"][0]):
    meta = raw["metadatas"][0][i] or {}
    dist = raw["distances"][0][i]
    pid = meta.get("parent_id") or meta.get("id") or f"row-{i}"
    score = base_score(dist) + month_boost(meta, doc, months_in_query)
    items.append(
        {
            "parent_id": pid,
            "score": score,
            "base_similarity": base_score(dist),
            "document": doc,
            "meta": meta,
        }
    )

best_by_parent = {}
for it in items:
    pid = it["parent_id"]
    if pid not in best_by_parent or it["score"] > best_by_parent[pid]["score"]:
        best_by_parent[pid] = it

reranked = sorted(best_by_parent.values(), key=lambda x: x["score"], reverse=True)[:5]

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
            "payment_method": m.get("payment_methods", ""),
            "similarity_percent": round(it["base_similarity"] * 100, 2),
            "score_after_boost": round(it["score"], 4),
            "description": it["document"],
        }
    )

print(json.dumps(output, ensure_ascii=False, indent=2))
