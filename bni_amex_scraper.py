import json
import re
import time
from urllib.parse import urljoin, urlparse, urlunparse, urlencode, parse_qsl

import requests
from bs4 import BeautifulSoup

BASE = "https://promo.bca.co.id"
CATEGORY_PATH = "/id/all/c/entertainment"
START_URL = urljoin(BASE, CATEGORY_PATH)
OUTFILE = "bca_entertainment_promos.json"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0 Safari/537.36"
    ),
    "Accept-Language": "id,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Connection": "close",
}

SESSION = requests.Session()
SESSION.headers.update(HEADERS)


def get(url, **kw):
    resp = SESSION.get(url, timeout=30, **kw)
    resp.raise_for_status()
    return resp


def normspace(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()


def make_page_url(url: str, page: int) -> str:
    """
    The site uses pagination (UI shows 1,2,3). It commonly works via ?page=N.
    We'll add/replace 'page' query param and stop when a page returns 0 cards.
    """
    u = urlparse(url)
    q = dict(parse_qsl(u.query, keep_blank_values=True))
    q["page"] = str(page)
    new_q = urlencode(q)
    return urlunparse((u.scheme, u.netloc, u.path, u.params, new_q, u.fragment))


def parse_cards(html: str):
    """
    Parse the category page for promo cards.
    Each card has a title <a>, a short description, and a 'Periode ...' line.
    Return list of dicts with title, periode_text, href (absolute).
    """
    soup = BeautifulSoup(html, "lxml")

    results = []
    # Cards are typically rendered as blocks with an h3/anchor and a "Periode ..." line nearby.
    for h3 in soup.select("h3, h2"):
        a = h3.find("a")
        if not a or not a.get("href"):
            continue
        title = normspace(a.get_text())
        href = urljoin(BASE, a["href"])

        # Look for a nearby line that starts with "Periode"
        periode_text = ""
        # The next siblings around the header often contain the “Periode …” text.
        for sib in h3.next_siblings:
            if getattr(sib, "get_text", None):
                t = normspace(sib.get_text())
                if t.lower().startswith("periode"):
                    periode_text = t
                    break

        # Extra guard: skip items that aren’t real promos (rare)
        if title and "/id/" in href:
            results.append({"title": title, "list_periode": periode_text, "url": href})

    # De-dup if the page marks things twice
    seen = set()
    uniq = []
    for r in results:
        if r["url"] in seen:
            continue
        seen.add(r["url"])
        uniq.append(r)

    return uniq


def extract_payment_methods(soup: BeautifulSoup):
    """
    Pull the chips under 'Bagi Pengguna' - often rendered as inline links or labels.
    """
    methods = []
    # Anchor or spans after "Bagi Pengguna" block
    # Example structure:
    # <h3>Bagi Pengguna</h3>
    # <a> Kartu Kredit </a> <a> Kartu Debit </a> <a> Virtual Account </a>
    header = None
    for node in soup.find_all(["h2", "h3"]):
        if normspace(node.get_text()).lower() == "bagi pengguna":
            header = node
            break

    if header:
        # collect siblings with links or spans
        for tag in header.find_all_next(["a", "span", "div"], limit=30):
            txt = normspace(tag.get_text())
            if not txt:
                continue
            # stop if we hit another section header
            if tag.name in ("h2", "h3") and tag is not header:
                break
            # Heuristic: only keep short-ish labels
            if len(txt) <= 40:
                # common labels: "Kartu Kredit", "Kartu Debit", "Virtual Account", "myBCA", "BCA mobile", "QRIS", "Sakuku"
                methods.append(txt)

    # Deduplicate and keep order
    out = []
    seen = set()
    for m in methods:
        if m not in seen:
            seen.add(m)
            out.append(m)
    return out


def extract_policy(soup: BeautifulSoup):
    """
    Grab bullet points under “Syarat & Ketentuan”.
    """
    items = []

    # Find the S&K header
    sk_header = None
    for node in soup.find_all(["h2", "h3"]):
        txt = normspace(node.get_text()).lower()
        if "syarat" in txt and "ketentuan" in txt:
            sk_header = node
            break

    if sk_header:
        # Get following <ul>/<ol> items or bullet-like paragraphs for a while
        # until the next section header
        for el in sk_header.find_all_next(["ul", "ol", "p", "li", "div"], limit=200):
            if el.name in ("h2", "h3") and el is not sk_header:
                break
            if el.name in ("ul", "ol"):
                for li in el.find_all("li"):
                    t = normspace(li.get_text(" "))
                    if t:
                        items.append(t)
            elif el.name == "p":
                # sometimes bullet items are just separate paragraphs
                t = normspace(el.get_text(" "))
                # skip trivial short lines that look like headings
                if t and len(t) > 15:
                    items.append(t)

    # Deduplicate while preserving order
    out = []
    seen = set()
    for it in items:
        if it not in seen:
            seen.add(it)
            out.append(it)
    return out


def extract_location_and_period(soup: BeautifulSoup):
    """
    Try to find 'Venue', 'Lokasi', and 'Periode promo' on the detail page.
    Also keep a best-effort 'period_detail' string.
    """
    location = ""
    periode_detail = ""

    # Look for 'Venue :' or 'Lokasi :' lines anywhere in body
    body_txt = normspace(soup.get_text(" "))
    m = re.search(
        r"(Venue|Lokasi)\s*:?\s*(.+?)(?=\s{2,}|Periode|Syarat|Bagikan|Promo Serupa|$)",
        body_txt,
        flags=re.I,
    )
    if m:
        location = normspace(m.group(2))

    # Prefer explicit "Periode promo: ..." if present
    m2 = re.search(r"Periode promo\s*:\s*([^\n\r]+)", body_txt, flags=re.I)
    if m2:
        periode_detail = normspace(m2.group(1))
    else:
        # Some pages show at the top “Periode 05 - 17 Agu 2025”
        m3 = re.search(r"\bPeriode\s+([0-9A-Za-z\-\s\.]+)", body_txt, flags=re.I)
        if m3:
            periode_detail = normspace(m3.group(1))

    return location, periode_detail


def parse_detail(url: str):
    """
    Open a promo detail page and extract desired fields.
    """
    try:
        r = get(url)
    except Exception as e:
        return {"error": f"detail fetch failed: {e}", "url": url}

    soup = BeautifulSoup(r.text, "lxml")

    title = (
        normspace(soup.find(["h1", "h2"]).get_text()) if soup.find(["h1", "h2"]) else ""
    )
    payment_methods = extract_payment_methods(soup)
    policy = extract_policy(soup)
    location, periode_detail = extract_location_and_period(soup)

    return {
        "title": title,
        "url": url,
        "payment_methods": payment_methods,
        "policy": policy,
        "location": location,
        "period": periode_detail,
    }


def crawl_category(start_url: str, max_pages: int = 50, sleep_sec: float = 0.8):
    """
    Iterate pages until no cards are found (or we hit max_pages).
    """
    all_rows = []
    page = 1
    while page <= max_pages:
        url = start_url if page == 1 else make_page_url(start_url, page)
        try:
            resp = get(url)
        except requests.HTTPError as e:
            # Stop if server returns 404/400 on deep pages
            if e.response is not None and e.response.status_code in (404, 400):
                break
            raise

        cards = parse_cards(resp.text)
        if not cards:
            # no more cards
            break

        for c in cards:
            detail = parse_detail(c["url"])
            # If detail period is empty, fall back to card period text (strip "Periode ")
            period_fallback = re.sub(
                r"^Periode\s*", "", c.get("list_periode", ""), flags=re.I
            ).strip()
            if not detail.get("period") and period_fallback:
                detail["period"] = period_fallback
            all_rows.append(detail)

            # be polite
            time.sleep(sleep_sec)

        page += 1
        time.sleep(sleep_sec)

    return all_rows


def main():
    data = crawl_category(START_URL)
    with open(OUTFILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(json.dumps(data, ensure_ascii=False, indent=2))
    print(f"\nSaved to {OUTFILE}", flush=True)


if __name__ == "__main__":
    main()
