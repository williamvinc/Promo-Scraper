import json
import re
import requests
from bs4 import BeautifulSoup

URL = "https://www.bni.co.id/creditcard/id-id/produk/produk-kartu-kredit-bni/bni-american-express-card"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0 Safari/537.36",
    "Accept-Language": "id-ID,id;q=0.9,en;q=0.8",
}


def norm_text(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").replace("\xa0", " ")).strip()


def cell_text(td):
    # Utamakan alt pada <img>
    img = td.find("img")
    if img and img.get("alt"):
        return norm_text(img.get("alt"))
    # Kalau ada link, ambil teksnya
    a = td.find("a")
    if a and a.get_text(strip=True):
        return norm_text(a.get_text(" "))
    # fallback: seluruh teks sel
    return norm_text(td.get_text(" "))


def parse_mekanisme(td):
    # Kumpulkan list mekanisme jika ada <li>
    items = []
    for li in td.find_all("li"):
        t = norm_text(li.get_text(" "))
        if t:
            items.append(t)
    if items:
        return items
    # Kalau tidak ada <li>, pecah berdasar <br>, bullet, titik-koma
    raw = td.decode_contents()
    # ganti <br> jadi delimiter
    raw = re.sub(r"(?i)<br\s*/?>", "|||", raw)
    text = norm_text(BeautifulSoup(raw, "html.parser").get_text(" "))
    parts = []
    for p in re.split(r"\|\|\||•|;|\n", text):
        p = norm_text(p)
        if p:
            parts.append(p)
    # Kalau setelah dipecah cuma 1 kalimat, tetap kembalikan 1 elemen
    return parts if parts else ([text] if text else [])


def find_target_table(soup):
    # Cari semua tabel, pilih yang headernya mengandung 4 kolom target (urutan fleksibel)
    target_cols = {"merchant", "program", "mekanisme", "periode"}
    for table in soup.find_all("table"):
        # Kumpulkan header (th) atau row pertama sebagai header
        headers = []
        thead = table.find("thead")
        if thead:
            for th in thead.find_all("th"):
                headers.append(norm_text(th.get_text(" ")).lower())
        else:
            first_tr = table.find("tr")
            if first_tr:
                ths = first_tr.find_all(["th", "td"])
                for th in ths:
                    headers.append(norm_text(th.get_text(" ")).lower())

        # Normalisasi nama header agar mudah dicocokkan
        normalized = set()
        for h in headers:
            h1 = re.sub(r"[^a-z]", "", h)
            normalized.add(h1)

        # mapping sederhana (merchant/program/mekanisme/periode harus ada semuanya)
        has = {
            "merchant": any("merchant" in h for h in headers),
            "program": any("program" in h for h in headers),
            "mekanisme": any("mekanisme" in h for h in headers),
            "periode": any("periode" in h for h in headers),
        }
        if all(has.values()):
            return table, headers
    return None, None


def build_column_index(headers):
    # Kembalikan index untuk merchant/program/mekanisme/periode berdasarkan header
    idx = {"merchant": None, "program": None, "mekanisme": None, "periode": None}
    for i, h in enumerate(headers):
        hl = h.lower()
        if "merchant" in hl and idx["merchant"] is None:
            idx["merchant"] = i
        elif "program" in hl and idx["program"] is None:
            idx["program"] = i
        elif "mekanisme" in hl and idx["mekanisme"] is None:
            idx["mekanisme"] = i
        elif "periode" in hl and idx["periode"] is None:
            idx["periode"] = i
    return idx


def scrape():
    resp = requests.get(URL, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    table, headers = find_target_table(soup)
    if not table:
        raise RuntimeError(
            "Tabel dengan header [Merchant, Program, Mekanisme, Periode] tidak ditemukan."
        )

    col_idx = build_column_index(headers)
    if not all(v is not None for v in col_idx.values()):
        # fallback: asumsikan urutan 4 kolom pertama
        col_idx = {"merchant": 0, "program": 1, "mekanisme": 2, "periode": 3}

    # Siapkan baris data: skip baris header
    rows = []
    for tr in table.find_all("tr"):
        tds = tr.find_all("td")
        if not tds:
            continue
        rows.append(tds)

    results = []
    for tds in rows:

        def get_td(i):
            return tds[i] if i is not None and i < len(tds) else None

        td_merchant = get_td(col_idx["merchant"])
        td_program = get_td(col_idx["program"])
        td_mekanisme = get_td(col_idx["mekanisme"])
        td_periode = get_td(col_idx["periode"])

        merchant = cell_text(td_merchant) if td_merchant else ""
        program = cell_text(td_program) if td_program else ""
        mekanisme = parse_mekanisme(td_mekanisme) if td_mekanisme else []
        periode = cell_text(td_periode) if td_periode else ""

        # filter baris kosong
        if not any([merchant, program, mekanisme, periode]):
            continue

        results.append(
            {
                "merchant": merchant,
                "program": program,
                "mekanisme": mekanisme,
                "periode": periode,
            }
        )

    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    scrape()
