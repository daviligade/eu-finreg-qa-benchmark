"""Align gold references with the Akoma Ntoso text of the two laws.

For every generated question that carries references, the script
  1. checks that each referenced article, paragraph and point exists in the structured text;
  2. scores how much of the gold answer each paragraph of the referenced article supports
     (weighted overlap of content words) and names the best supported paragraph.

Usage:
    python -m dataset.align_references                 # check, print a summary
    python -m dataset.align_references --csv out.csv   # also write the per-reference scores

The exit code is 1 when a reference does not exist in the structured text.
"""
import argparse, collections, csv, json, math, re, sys
import xml.etree.ElementTree as ET
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"
NS = "{http://docs.oasis-open.org/legaldocml/ns/akn/3.0}"
CELEX = {"MIFID2_L1": "32014L0065", "SFDR_L1": "32019R2088"}
STOP = set("""a an the of to in on for and or by with as at from that this these those is are be been was were shall may
must can will not no any all each such which who whom whose it its their there where when if than then also other into under
over between within without per via including include includes only more most one two three paragraph paragraphs article
articles point points referred accordance provided provide provides member state states""".split())


def _txt(el):
    return " ".join("".join(el.itertext()).split())


def load_law(instrument):
    """{article: [(paragraph number, text, point letters)]}; an article without numbered paragraphs has one entry ''."""
    root = ET.parse(DATA / "akn" / f"{CELEX[instrument]}.akn").getroot()
    parent = {c: p for p in root.iter() for c in p}

    def quoted(e):
        while e in parent:
            e = parent[e]
            if e.tag.replace(NS, "") in ("mod", "quotedStructure"):
                return True
        return False

    arts = {}
    for a in root.iter(NS + "article"):
        if quoted(a):
            continue
        num = re.search(r"(\d+[a-z]?)", _txt(a.find(NS + "num"))).group(1)
        paras = []
        for p in a.findall(NS + "paragraph"):
            n = p.find(NS + "num")
            points = {re.sub(r"[^a-z]", "", _txt(x)) for x in p.iter(NS + "num") if x is not n}
            points |= set(re.findall(r"(?:^|[\s:;,])\(([a-z])\)\s", _txt(p)))
            paras.append((re.sub(r"[^0-9a-z]", "", _txt(n)) if n is not None else "", _txt(p), points))
        arts[num] = paras or [("", _txt(a), set())]
    return arts


def _tokens(text):
    out = []
    for w in re.findall(r"[a-z][a-z\-]+|\d+", text.lower()):
        if w in STOP or len(w) < 3:
            continue
        for suf in ("ations", "ation", "ments", "ment", "ings", "ing", "ies", "es", "s", "ed"):
            if w.endswith(suf) and len(w) - len(suf) >= 4:
                w = w[: -len(suf)]
                break
        out.append(w)
    return out


def support(answer, paragraphs):
    """For each paragraph, the best share of one answer sentence's weighted content words found in it."""
    docs = [set(_tokens(p[1])) for p in paragraphs]
    df = collections.Counter(w for d in docs for w in d)
    idf = lambda w: math.log(1 + len(docs) / (1 + df.get(w, 0))) + 0.3
    sents = [s for s in re.split(r"(?<=[.;:])\s+", answer) if len(_tokens(s)) >= 3] or [answer]
    scores = []
    for d in docs:
        best = 0.0
        for s in sents:
            ts = set(_tokens(s))
            den = sum(idf(w) for w in ts) or 1
            best = max(best, sum(idf(w) for w in ts if w in d) / den)
        scores.append(round(best, 3))
    return scores


def parse_reference(s):
    m = re.search(r"(\d+[a-z]?)\s*(?:\(\s*(\d+[a-z]?)\s*\))?\s*(?:\(\s*([a-z])\s*\))?", str(s))
    return (m.group(1), m.group(2) or "", m.group(3) or "") if m else None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--csv", help="write per-reference scores to this file")
    args = ap.parse_args(argv)
    law = {i: load_law(i) for i in CELEX}
    rows, missing, n_items = [], [], 0
    for line in open(DATA / "synthetic.jsonl", encoding="utf-8"):
        q = json.loads(line)
        if not q["gold_article_pinpoints"]:
            continue
        n_items += 1
        instruments = q["seed_instrument"].split("|")
        seeds = dict(zip(str(q["seed_article"]).split("|"), instruments))
        for ref in q["gold_article_pinpoints"]:
            art, para, point = parse_reference(ref)
            inst = seeds.get(art) or next((i for i in instruments if art in law[i]), None)
            paras = law[inst].get(art) if inst else None
            numbers = [p[0] for p in paras] if paras else []
            exists = bool(paras) and (not para or para in numbers)
            if exists and point:
                exists = point in paras[numbers.index(para)][2]
            best, score = "", ""
            if paras and para:
                s = support(q["gold_answer"], paras)
                k = max(range(len(s)), key=s.__getitem__)
                best = numbers[k]
                score = s[numbers.index(para)] if para in numbers else ""
            rows.append({"id": q["id"], "reference": ref, "instrument": inst or "", "exists": exists,
                         "support": score, "best_supported_paragraph": best})
            if not exists:
                missing.append((q["id"], ref))
    print(f"{n_items} questions, {len(rows)} references, {len(missing)} not found in the structured text")
    for qid, ref in missing:
        print("  not found:", qid, ref)
    with_para = [r for r in rows if r["best_supported_paragraph"]]
    agree = sum(parse_reference(r["reference"])[1] == r["best_supported_paragraph"] for r in with_para)
    print(f"{agree} of {len(with_para)} paragraph references name the paragraph with the highest overlap")
    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
