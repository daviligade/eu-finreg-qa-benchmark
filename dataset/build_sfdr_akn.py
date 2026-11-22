"""Rebuild data/akn/32019R2088.akn from the official SFDR PDF.


USAGE
  python -m dataset.build_sfdr_akn            # writes data/akn/32019R2088.akn
  python -m dataset.build_sfdr_akn --dry-run  # parse and report, write nothing

The old file is kept next to the new one with the suffix .skeleton-bak.
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

from lxml import etree
from pypdf import PdfReader

AKN_NS = "http://docs.oasis-open.org/legaldocml/ns/akn/3.0"

PDF_PATH = Path("data/pdfs/CELEX_32019R2088_EN_TXT.pdf")
OUT_PATH = Path("data/akn/32019R2088.akn")
MIFID_AKN = Path("data/akn/32014L0065.akn")

# Words that stand alone so often that joining them to the next word
# would do more harm than good ("of ten" must not become "often").
_NO_JOIN_LEFT = frozenset("""
a i an as at be by do if in is it no of on or so to we and are for had
has her his its may not one out per ten two the via was who why how all
can will this that from have been were such each than then them they
when where which their there these those under after about above
""".split())


def _reference_vocab() -> set:
    """Clean words to vote with: the MiFID AKN plus the old SFDR file."""
    vocab: set = set()
    for path in (MIFID_AKN, OUT_PATH):
        if not path.exists():
            continue
        try:
            tree = etree.parse(str(path))
        except Exception:
            continue
        text = " ".join(tree.getroot().itertext())
        vocab.update(re.findall(r"[a-z]{3,}", text.lower()))
    # The extraction above keeps words of 3+ letters, so legitimate
    # 2-letter words could never win a join vote ("b y" -> "by").
    vocab.update({"by", "of", "to", "in", "on", "at", "or", "be", "an",
                  "as", "is", "it", "no", "up", "if"})
    return vocab


def _dekern(text: str, vocab: set) -> str:
    """Join split word fragments when the joined form is a known word."""
    for ch in (" ", " ", " ", " ", " "):
        text = text.replace(ch, " ")

    def join_line(line: str) -> str:
        toks = re.sub(r" {2,}", " ", line).split(" ")
        out = []
        i = 0
        while i < len(toks):
            cur = toks[i]
            # Absorb following fragments into cur, chaining while cur
            # stays pure letters and each join is vocabulary-approved.
            while i + 1 < len(toks) and re.fullmatch(r"[A-Za-z]{1,12}", cur):
                if cur.lower() in _NO_JOIN_LEFT:
                    break
                m = re.fullmatch(r"([a-z]{1,12})([^A-Za-z]*)", toks[i + 1])
                if not m:
                    break
                frag, tail = m.group(1), m.group(2)
                if (cur + frag).lower() in vocab:
                    cur = cur + frag + tail
                    i += 1
                    if tail:
                        break
                    continue
                # Two fragments were not a word; try three, for splits
                # like "inter mediar ies" and "co v ered".
                if not tail and i + 2 < len(toks):
                    m2 = re.fullmatch(r"([a-z]{1,12})([^A-Za-z]*)",
                                      toks[i + 2])
                    if m2 and (cur + frag + m2.group(1)).lower() in vocab:
                        cur = cur + frag + m2.group(1) + m2.group(2)
                        i += 2
                        if m2.group(2):
                            break
                        continue
                break
            out.append(cur)
            i += 1
        return " ".join(out)

    for _ in range(3):
        new = "\n".join(join_line(l) for l in text.split("\n"))
        if new == text:
            break
        text = new
    return text


_HEADER_RE = re.compile(
    r"^\s*EN\s*$"
    r"|Offi\s?cial\s+Jour\s?nal\s+of\s+the\s+European\s+Uni\s?on"
    r"|^\s*L\s*317/\d+\s*$"
    r"|^\s*\d{1,2}\.\d{1,2}\.\d{4}\s*$",
    re.IGNORECASE)


def _clean_lines(block: str) -> list:
    """Drop running headers/footers and page furniture from a block."""
    out = []
    for line in block.splitlines():
        s = line.strip()
        if not s:
            continue
        if _HEADER_RE.search(s):
            continue
        # Footer lines mix journal name, page and date with heavy spacing.
        if "Jour nal" in s or "Journal of the European" in s:
            continue
        out.append(s)
    return out


def parse_articles(text: str) -> list:
    """Cut the PDF text into (number, heading, [(para_num, para_text)])."""
    marks = [(int(m.group(1)), m.start(), m.end())
             for m in re.finditer(r"\nAr\s?ticle\s+(\d+)\s*\n", text)]
    articles = []
    for i, (num, _s, e) in enumerate(marks):
        end = marks[i + 1][1] if i + 1 < len(marks) else len(text)
        block = text[e:end]
        # The regulation's operative part ends before the signature block.
        block = re.split(r"\nDone at [A-Z]", block)[0]
        lines = _clean_lines(block)
        if not lines:
            continue
        # Heading: first line; allow one continuation line when the next
        # line is not a numbered paragraph and the first looks unfinished.
        heading = lines[0]
        body_start = 1
        if (len(lines) > 1
                and not re.match(r"^\d{1,2}\.\s", lines[1])
                and len(heading) < 90
                and not heading.endswith(".")
                and len(lines[1]) < 80
                and not lines[1].startswith("(")
                and re.match(r"^[a-z]", lines[1])):
            heading = heading + " " + lines[1]
            body_start = 2
        body = "\n".join(lines[body_start:])
        # Paragraphs: split on "N." at line start. Text before the first
        # number (if any) becomes one unnumbered paragraph.
        parts = re.split(r"(?m)^(\d{1,2})\.\s", body)
        paras = []
        lead = " ".join(parts[0].split())
        if lead:
            paras.append(("", lead))
        for j in range(1, len(parts), 2):
            pnum = parts[j]
            ptext = " ".join(parts[j + 1].split()) if j + 1 < len(parts) else ""
            if ptext:
                paras.append((pnum, ptext))
        articles.append((num, heading, paras))
    return articles


def build_xml(articles: list) -> etree._ElementTree:
    nsmap = {None: AKN_NS}
    E = lambda tag, parent=None: (
        etree.SubElement(parent, f"{{{AKN_NS}}}{tag}") if parent is not None
        else etree.Element(f"{{{AKN_NS}}}{tag}", nsmap=nsmap))
    root = E("akomaNtoso")
    act = etree.SubElement(root, f"{{{AKN_NS}}}act")
    body = etree.SubElement(act, f"{{{AKN_NS}}}body")
    for num, heading, paras in articles:
        art = etree.SubElement(body, f"{{{AKN_NS}}}article")
        art.set("eId", f"article-{num}")
        n = etree.SubElement(art, f"{{{AKN_NS}}}num")
        n.text = f"Article {num}"
        h = etree.SubElement(art, f"{{{AKN_NS}}}heading")
        h.text = heading
        for pnum, ptext in paras:
            para = etree.SubElement(art, f"{{{AKN_NS}}}paragraph")
            para.set("eId", f"article-{num}-{pnum or 'intro'}")
            if pnum:
                pn = etree.SubElement(para, f"{{{AKN_NS}}}num")
                pn.text = f"{pnum}."
            content = etree.SubElement(para, f"{{{AKN_NS}}}content")
            p = etree.SubElement(content, f"{{{AKN_NS}}}p")
            p.text = ptext
    return etree.ElementTree(root)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    reader = PdfReader(str(PDF_PATH))
    raw = "\n".join(page.extract_text() or "" for page in reader.pages)
    vocab = _reference_vocab()
    print(f"reference vocabulary: {len(vocab)} words")
    text = _dekern(raw, vocab)
    articles = parse_articles(text)
    print(f"parsed {len(articles)} articles")
    for num, heading, paras in articles:
        total = sum(len(t) for _, t in paras)
        print(f"  Art {num:>2d}  {total:6d} chars  {len(paras):2d} paras  {heading[:60]}")
    if args.dry_run:
        return 0
    if OUT_PATH.exists():
        bak = OUT_PATH.with_suffix(".akn.skeleton-bak")
        if not bak.exists():
            shutil.copy2(OUT_PATH, bak)
            print(f"old file kept at {bak}")
    tree = build_xml(articles)
    tree.write(str(OUT_PATH), xml_declaration=True, encoding="UTF-8",
               pretty_print=True)
    print(f"wrote {OUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
