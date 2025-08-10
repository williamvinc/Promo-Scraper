import json
import re
import time
import chromadb
import os
from typing import List, Dict
from datetime import datetime, timezone
from urllib.parse import urljoin, urlparse, urlunparse, urlencode, parse_qsl

import requests
import hashlib
from bs4 import BeautifulSoup, NavigableString, Tag
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

BASE = "https://promo.bca.co.id"
# CATEGORY_PATH = "/id/all/c/telco?order=DESC&sort=DATE"
CATEGORY_PATH = "/id/all?order=DESC&sort=RELEVANCE"
START_URL = urljoin(BASE, CATEGORY_PATH)
OUTFILE = "website/bca/bca.json"

CHROMA_HOST = "110.239.80.161"
CHROMA_PORT = 8000
COLLECTION_NAME = "promo_collection"

CHUNK_MAX_CHARS = 1200
CHUNK_OVERLAP = 150
BATCH_SIZE = 96
DROP_OLD_COLLECTION = True

MONTH_ID = r"(?:Jan(?:uari)?|Feb(?:ruari)?|Mar(?:et)?|Apr(?:il)?|Mei|Jun(?:i)?|Jul(?:i)?|Agu(?:stus)?|Sep(?:tember)?|Okt(?:ober)?|Nov(?:ember)?|Des(?:ember)?)"
DATE_RE = rf"(\d{{1,2}}\s+{MONTH_ID}\s+\d{{4}})"
DATE_RANGE_RE = rf"{DATE_RE}\s*[-–]\s*{DATE_RE}"

HEADERS = {"User-Agent": "Mozilla/5.0", "Accept-Language": "id,en;q=0.9"}
SESSION = requests.Session()
SESSION.headers.update(HEADERS)


def make_embed_text(meta: dict, chunk: str) -> str:
    key_order = [
        "title",
        "period",
        "category",
        "bank",
        "payment_methods",
        "scrape_date",
        "url",
        "parent_id",
        "chunk_index",
    ]
    m = {k: ("" if v is None else v) for k, v in (meta or {}).items()}

    if isinstance(m.get("payment_methods"), (list, tuple)):
        m["payment_methods"] = ", ".join(m["payment_methods"])

    lines = []
    seen = set()

    for k in key_order:
        if k in m and k != "id":
            val = str(m[k]).strip()
            if val:
                lines.append(f"{k.title().replace('_',' ')}: {val}")
            seen.add(k)

    for k, v in m.items():
        if k in seen or k == "id":
            continue
        val = str(v).strip()
        if val:
            lines.append(f"{k.title().replace('_',' ')}: {val}")

    header = "\n".join(lines)
    body = (chunk or "").strip()

    if header and body:
        return f"{header}\n\n{body}"
    return header or body


def normspace(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()


def get(url, **kw):
    resp = SESSION.get(url, timeout=30, **kw)
    resp.raise_for_status()
    return resp


def make_page_url(url: str, page: int) -> str:
    u = urlparse(url)
    q = dict(parse_qsl(u.query, keep_blank_values=True))
    q["page"] = str(page)
    return urlunparse((u.scheme, u.netloc, u.path, u.params, urlencode(q), u.fragment))


def batched(iterable, batch_size: int):
    batch = []
    for x in iterable:
        batch.append(x)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def chunk_text(
    text: str, max_chars: int = CHUNK_MAX_CHARS, overlap: int = CHUNK_OVERLAP
) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = max(0, end - overlap)
    return chunks


def parse_cards(html: str):
    soup = BeautifulSoup(html, "lxml")
    results = []
    for h in soup.select("h3, h2"):
        a = h.find("a")
        if not a or not a.get("href"):
            continue
        title = normspace(a.get_text())
        href = urljoin(BASE, a["href"])
        periode_text = ""
        for sib in h.next_siblings:
            if getattr(sib, "get_text", None):
                t = normspace(sib.get_text())
                if t.lower().startswith("periode"):
                    periode_text = t
                    break
        if title and "/id/" in href:
            results.append({"title": title, "list_periode": periode_text, "url": href})
    seen, uniq = set(), []
    for r in results:
        if r["url"] not in seen:
            seen.add(r["url"])
            uniq.append(r)
    return uniq


def extract_payment_methods(soup: BeautifulSoup):
    methods = []
    header = soup.find(
        lambda t: isinstance(t, Tag)
        and t.name in ("h2", "h3")
        and normspace(t.get_text()).lower() == "bagi pengguna"
    )
    if header:
        container = header.find_next(
            lambda t: isinstance(t, Tag) and t.name in ("div", "ul")
        )
        if container:
            for tag in container.find_all(["a", "span", "li"], recursive=True):
                txt = normspace(tag.get_text())
                if txt and len(txt) <= 40:
                    methods.append(txt)
    out, seen = [], set()
    for m in methods:
        if m not in seen:
            seen.add(m)
            out.append(m)
    return out


def extract_period(soup: BeautifulSoup) -> str:
    body = soup.get_text(" ", strip=True)
    match = re.search(DATE_RANGE_RE, body, flags=re.I)
    if match:
        return f"{match.group(1)} - {match.group(2)}"
    match = re.search(DATE_RE, body, flags=re.I)
    if match:
        return match.group(1)
    return ""


def extract_description(soup: BeautifulSoup) -> str:
    paras = []
    for el in soup.find_all(["p", "li"], limit=300):
        t = normspace(el.get_text(" "))
        if t and len(t) > 2:
            paras.append(t)
    return "\n".join(paras[:50]).strip()


def extract_category(soup: BeautifulSoup) -> str:
    def is_breadcrumb_container(tag: Tag) -> bool:
        classes = " ".join(tag.get("class", [])).lower()
        text = normspace(tag.get_text(" "))
        return ("bread" in classes) or ("Home" in text and "Promo BCA" in text)

    bc = soup.find(is_breadcrumb_container)
    if not bc:
        promo_node = soup.find(string=re.compile(r"^\s*Promo BCA\s*$", re.I))
        if promo_node and isinstance(promo_node, NavigableString):
            bc = promo_node.parent

    if not bc:
        return ""

    raw_tokens = []
    for el in bc.find_all(["a", "span", "li"], recursive=True):
        t = normspace(el.get_text(" "))
        if t:
            raw_tokens.append(t)

    seen, tokens = set(), []
    for t in raw_tokens:
        if t not in seen:
            seen.add(t)
            tokens.append(t)

    for key in ("Promo BCA", "Promo"):
        if key in tokens:
            idx = tokens.index(key)
            if idx + 1 < len(tokens):
                cat = tokens[idx + 1]
                if cat.lower() not in ("home", "beranda"):
                    return cat
    return ""


def parse_detail(url: str):
    try:
        r = get(url)
    except Exception as e:
        return {"error": f"detail fetch failed: {e}", "url": url}
    soup = BeautifulSoup(r.text, "lxml")
    return {
        "title": (
            normspace(soup.find(["h1", "h2"]).get_text())
            if soup.find(["h1", "h2"])
            else ""
        ),
        "url": url,
        "payment_methods": extract_payment_methods(soup),
        "period": extract_period(soup),
        "description": extract_description(soup),
        "category": extract_category(soup),
    }


def crawl_category(start_url: str, max_pages: int = 100, sleep_sec: float = 0.8):
    all_rows, page = [], 1
    while page <= max_pages:
        url = start_url if page == 1 else make_page_url(start_url, page)
        print(f"\n[INFO] Scraping page {page}: {url}")
        try:
            resp = get(url)
        except requests.HTTPError as e:
            if e.response is not None and e.response.status_code in (404, 400):
                break
            raise
        cards = parse_cards(resp.text)
        if not cards:
            break
        for idx, c in enumerate(cards, start=1):
            print(f"  [INFO] ({page}-{idx}/{len(cards)}) {c['title']}")
            detail = parse_detail(c["url"])
            detail["id"] = hashlib.sha256(c["url"].encode("utf-8")).hexdigest()
            period_fallback = re.sub(
                r"^Periode\s*", "", c.get("list_periode", ""), flags=re.I
            ).strip()
            if not detail.get("period") and period_fallback:
                detail["period"] = period_fallback
            detail["scrape_date"] = datetime.now(timezone.utc).isoformat()
            detail["bank"] = "BCA"
            detail["index"] = len(all_rows)
            all_rows.append(detail)
            time.sleep(sleep_sec)
        page += 1
        time.sleep(sleep_sec)
    return all_rows


def main():
    data = crawl_category(START_URL)
    os.makedirs(os.path.dirname(OUTFILE), exist_ok=True)
    with open(OUTFILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"[INFO] Saved {len(data)} records to {OUTFILE}")

    client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
    if DROP_OLD_COLLECTION:
        try:
            client.delete_collection(COLLECTION_NAME)
        except:
            pass
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    model = SentenceTransformer("LazarusNLP/all-indo-e5-small-v4")

    existing_docs = collection.get(ids=None, include=["metadatas"])
    existing_ids = existing_docs["ids"]

    existing_parent_ids = set()
    for meta_list in existing_docs["metadatas"]:
        for meta in meta_list:
            pid = meta.get("parent_id")
            if pid:
                existing_parent_ids.add(pid)

    scraped_parent_ids = {item["id"] for item in data}

    parent_ids_to_delete = existing_parent_ids - scraped_parent_ids
    if parent_ids_to_delete:
        ids_to_delete = [
            doc_id
            for doc_id, metas in zip(existing_docs["ids"], existing_docs["metadatas"])
            if metas.get("parent_id") in parent_ids_to_delete
        ]
        if ids_to_delete:
            collection.delete(ids=ids_to_delete)
            print(f"[DELETE] {len(ids_to_delete)} chunks deleted (promo removed)")

    total_chunks = 0
    for item in data:
        parent_id = item["id"]

        ids_to_delete = [
            doc_id
            for doc_id, metas in zip(existing_docs["ids"], existing_docs["metadatas"])
            if metas.get("parent_id") == parent_id
        ]
        if ids_to_delete:
            collection.delete(ids=ids_to_delete)
            print(f"[DELETE] {len(ids_to_delete)} old chunks for promo {parent_id}")

        chunks = chunk_text(item.get("description", ""))
        chunk_records = []
        for j, ch in enumerate(chunks):
            chunk_records.append(
                {
                    "id": f"{parent_id}#chunk-{j}",
                    "document": ch,
                    "metadata": {
                        "parent_id": parent_id,
                        "title": item["title"],
                        "url": item["url"],
                        "payment_methods": ", ".join(item.get("payment_methods", [])),
                        "period": item.get("period", ""),
                        "category": item.get("category", ""),
                        "scrape_date": item.get("scrape_date", ""),
                        "bank": item.get("bank", "BCA"),
                        "chunk_index": j,
                    },
                }
            )

        ids = [rec["id"] for rec in chunk_records]
        docs_original = [rec["document"] for rec in chunk_records]
        metas = [rec["metadata"] for rec in chunk_records]

        docs_aug = [make_embed_text(m, d) for m, d in zip(metas, docs_original)]

        embeds = model.encode(docs_aug, convert_to_numpy=True).tolist()
        collection.upsert(
            ids=ids, documents=docs_original, metadatas=metas, embeddings=embeds
        )

        total_chunks += len(chunk_records)
        print(f"[UPSERT] {parent_id} → {len(chunk_records)} chunks")

    print(f"Total chunks upserted: {total_chunks}")
    print("Total items in collection:", collection.count())


if __name__ == "__main__":
    main()
