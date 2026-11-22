# A benchmark for question answering over EU financial regulation

Benchmark, generation pipeline, audit script and evaluation harness for question answering over
MiFID II (Directive 2014/65/EU) and SFDR (Regulation (EU) 2019/2088).

## Contents

| Path | What |
|---|---|
| `data/esma_qa.jsonl` | 151 real questions with the regulator's published answer as gold answer, references to MiFID II (`gold_article_pinpoints`), references to other legal documents (`other_instrument_references`) and source pages |
| `data/synthetic.jsonl` | 223 generated questions with gold answers and article references |
| `data/synthetic_audit.jsonl` | Scores given to each generated question by a second model family |
| `data/akn/` | Akoma Ntoso files of the two laws |
| `dataset/` | Generation pipeline, audit script, reference alignment script, extractor for the real questions, Akoma Ntoso builder for SFDR |
| `evaluation/` | Harness, deterministic metrics, judge cache, paired statistics, table builder, unit tests |
| `baselines/` | Two baseline retrieval systems: B0 (dense) and B1 (dense + BM25, rank fusion) |
| `wrappers/_cite_parse.py` | Citation parser used by the metrics |
| `manifest.yaml` | Configuration: models, seed, sample sizes, statistics |

The generated set holds seven kinds of questions (field `intent_hint`): `FACTUAL_QA` (77), `OFF_TOPIC`
refusal traps (60), `SUMMARIZATION` (33), `LEGAL_ADVICE` (21), `COMPARISON` across the two laws (16),
`CONVERSATIONAL` (8) and `META` (8).

## Gold references

Gold references are aligned with the Akoma Ntoso text of the two laws. Every reference in
`data/synthetic.jsonl` names an article, paragraph or point that exists in `data/akn/`; a question that
asks for a summary of a whole article cites the article. `python -m dataset.align_references` checks
this and reports, for each reference, how much of the gold answer the referenced paragraph supports.
Where this overlap did not single out one paragraph, a language model (Claude Fable 5.2) read the item
against the article and its reading is used. An expert review of the references is ongoing.

For the real questions, each reference number in the regulator's answer is assigned to the legal
document it belongs to. Only references to MiFID II are in `gold_article_pinpoints` and are scored;
references to delegated acts and other documents outside the corpus are in `other_instrument_references`.

Use `gold_article_pinpoints` as the reference of record: paragraph numbers that appear inside the text
of a generated gold answer come from the generator and are not aligned. The scores in
`data/synthetic_audit.jsonl` were given before the alignment step.

## Install and check

```bash
pip install -r requirements.txt
python -m pytest -q evaluation wrappers
```

## Run the baselines

The harness needs [Ollama](https://ollama.com) with the three models named in `manifest.yaml`, and the
source PDFs, which are not redistributed here:

```bash
bash dataset/download_esma.sh                      # fetches the PDFs into data/pdfs/
python -m evaluation.harness --manifest manifest.yaml --conference --systems B0,B1
python -m evaluation.paper_tables runs/<run_dir>
```

`--conference` draws 50 real and 100 generated questions with the seed in `manifest.yaml`; `--pilot`
draws 10 and 20. Add `--skip-ragas` to compute only the deterministic metrics. Each run writes its
answers, per-question metrics, summary and paired comparisons to `runs/<run_dir>/`.

## Evaluate another system

Add a module that exposes `build(manifest, corpus)` and returns an object with
`answer(question, intent=None)`; the expected fields are described at the top of
`evaluation/harness.py`. Register it under `systems:` in `manifest.yaml`. In the table builder the
ids `V1` and `V2` are reserved for a multi-query system (MQ) and a system that indexes Akoma Ntoso
article units (AKN); these two systems are not part of this repository.

## Metrics

Six deterministic metrics are defined in `evaluation/metrics.py`: citation validity (CVR), article
accuracy at article and at pinpoint level (APA_article, APA_pinpoint), multi-instrument coverage (MIC),
refusal correctness (RC) and citation density (CD). APA_pinpoint is computed only where the gold reference
names a paragraph. An answer counts as a refusal for RC when the
system flags it as one or when it matches the phrase list `_REFUSAL_MARKERS` in the same file.

## How to cite

```bibtex
@inproceedings{liga2026benchmark,
  title     = {Building and Validating a Benchmark for Question Answering over {EU} Financial Regulation: Early Results},
  author    = {Liga, Davide and Yu, Liuwen and Arora, Ankit and Najjar, Amro},
  booktitle = {Proceedings of the 1st International Workshop on Multi-Agent and Agentic Systems (MASO 2026)},
  address   = {Kumamoto, Japan},
  year      = {2026}
}
```

## License and sources

Code: MIT (see `LICENSE`). Generated questions, gold answers and annotations: CC BY 4.0.

The real questions and answers are extracted from *Questions and Answers on MiFID II and MiFIR investor
protection and intermediaries topics* (ESMA35-43-349, version of 15 December 2023), © European Securities and Markets Authority,
available free of charge at <https://www.esma.europa.eu>. They are reproduced with the source
acknowledged; ESMA does not endorse this publication. The legal texts are © European Union, source:
<https://eur-lex.europa.eu>; only the Official Journal of the European Union is authentic. The Akoma Ntoso
file of MiFID II is the version as adopted published by <https://www.legislation.gov.uk>, whose content
derived from EUR-Lex is reused under Commission Decision 2011/833/EU; the SFDR file was built from the EUR-Lex text with `dataset/build_sfdr_akn.py`.
