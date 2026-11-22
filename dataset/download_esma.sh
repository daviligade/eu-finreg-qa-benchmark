#!/usr/bin/env bash
# Fetch the ESMA Q&A / Guidelines and the EUR-Lex L1 instruments into data/pdfs/.
# Safe to re-run: an existing non-empty file is skipped.
#
# EUR-Lex notes:
#   - The PDF endpoint redirects, requires a browser-style User-Agent,
#     and emits the file with an arbitrary name. We pin the output name
#     to the canonical "CELEX_<id>_EN_TXT.pdf" form the harness expects.
#   - We try TWO URL forms per CELEX id and accept the first that yields
#     a real PDF (file starts with "%PDF").

set -uo pipefail
cd "$(dirname "$0")/.."

mkdir -p data/pdfs

UA='Mozilla/5.0 (Macintosh; Intel Mac OS X 14_5) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.5 Safari/605.1.15'

is_pdf() {
  # Returns 0 if the file's first bytes are "%PDF".
  [ -s "$1" ] && head -c 4 "$1" | grep -q '%PDF'
}

fetch_to() {
  local url="$1" out="$2"
  echo "[get]  $(basename "$out")  <- ${url}"
  curl -fL --retry 3 --retry-delay 2 \
       -A "$UA" -H 'Accept: application/pdf,*/*' \
       -o "$out" "$url"
}

fetch_pdf() {
  # fetch_pdf <out_path> <url1> [url2 ...]
  local out="$1"; shift
  if is_pdf "$out"; then
    echo "[skip] $(basename "$out") (already a valid PDF)"
    return 0
  fi
  for url in "$@"; do
    rm -f "$out"
    if fetch_to "$url" "$out"; then
      if is_pdf "$out"; then
        echo "[ok]   $(basename "$out") ($(wc -c <"$out") bytes)"
        return 0
      fi
      echo "[warn] $(basename "$out") downloaded but is not a PDF; trying next URL"
    fi
  done
  rm -f "$out"
  echo "[FAIL] $(basename "$out") — try downloading manually from EUR-Lex."
  return 1
}

#################################
# 1) ESMA evaluation PDFs
#################################
fetch_pdf "data/pdfs/esma35-43-349_mifid_ii_qas_on_investor_protection_topics.pdf" \
  "https://www.esma.europa.eu/sites/default/files/library/esma35-43-349_mifid_ii_qas_on_investor_protection_topics.pdf"

fetch_pdf "data/pdfs/ESMA35-43-3172_Guidelines_on_certain_aspects_of_the_MiFID_II_suitability_requirements.pdf" \
  "https://www.esma.europa.eu/sites/default/files/2023-04/ESMA35-43-3172_Guidelines_on_certain_aspects_of_the_MiFID_II_suitability_requirements.pdf"

#################################
# 2) EUR-Lex L1 instruments
#################################
# MiFID II — Directive 2014/65/EU (CELEX 32014L0065)
fetch_pdf "data/pdfs/CELEX_32014L0065_EN_TXT.pdf" \
  "https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=CELEX:32014L0065&from=EN" \
  "https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=CELEX:32014L0065" \
  "https://eur-lex.europa.eu/resource.html?uri=cellar:f482f0aa-8a06-11e7-bca7-01aa75ed71a1.0006.02/DOC_1&format=PDF"

# SFDR — Regulation (EU) 2019/2088 (CELEX 32019R2088)
fetch_pdf "data/pdfs/CELEX_32019R2088_EN_TXT.pdf" \
  "https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=CELEX:32019R2088&from=EN" \
  "https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=CELEX:32019R2088" \
  "https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=CELEX:02019R2088-20200712"

echo
echo "Files in data/pdfs:"
ls -lh data/pdfs | sed 's/^/  /'

# Final report ----------------------------------------------------------------
missing=0
for f in \
    data/pdfs/esma35-43-349_mifid_ii_qas_on_investor_protection_topics.pdf \
    data/pdfs/ESMA35-43-3172_Guidelines_on_certain_aspects_of_the_MiFID_II_suitability_requirements.pdf \
    data/pdfs/CELEX_32014L0065_EN_TXT.pdf \
    data/pdfs/CELEX_32019R2088_EN_TXT.pdf
do
  if ! is_pdf "$f"; then
    missing=1
    echo "  MISSING: $f"
  fi
done

if [ "$missing" -ne 0 ]; then
  cat <<'EOF'

One or more EUR-Lex PDFs could not be auto-downloaded. EUR-Lex sometimes
geo-blocks or rate-limits scripted access. Please download manually:

  MiFID II (Directive 2014/65/EU)
    https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32014L0065
    -> click the PDF (EN) link, save as:
       data/pdfs/CELEX_32014L0065_EN_TXT.pdf

  SFDR (Regulation (EU) 2019/2088)
    https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32019R2088
    -> click the PDF (EN) link, save as:
       data/pdfs/CELEX_32019R2088_EN_TXT.pdf

EOF
  exit 1
fi
echo "All PDFs OK."
