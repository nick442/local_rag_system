Absolutely—let’s redesign your **performance‑focused RAG experiments** so they are tightly grounded in the most relevant literature and fill clear gaps, while staying feasible on a **Mac mini M4, 16 GB RAM**. Below you’ll find:

1. a concise **research program** (6 core experiments + 2 optional add‑ons),
2. **why each matters** in the literature (what’s known vs. what’s missing),
3. **exact procedures** (datasets/corpora, query sets, metrics, statistics, and ablations), and
4. an **execution plan** you can plug into your modular agent.

---

## Executive overview (what you’ll contribute)

* **Practical gap**: Most RAG papers evaluate on servers/GPUs or at web scale. We will map the **Pareto frontier of quality vs. latency/memory** for a *local* RAG on commodity Apple Silicon, under a rigorous, reproducible protocol (BEIR‑style retrieval metrics, paired significance tests).
* **Scientific gap**: (a) **Chunking** (fixed vs. semantic vs. *late chunking*) has only recently seen systematic study; behavior for small dense models and CPU/MPS inference at low memory bounds is underexplored. (b) **Hybrid retrieval** (BM25+dense) is known to help, but **how to weight/fuse** on small setups and per‑query is unsettled. (c) **LLM‑based query expansion** (HyDE/query2doc) is effective, yet **leakage and corpus steering** remain open issues. (d) **Lightweight rerankers** on CPU (MiniLM or BGE‑v2‑m3) vs. cost/latency trade‑offs under local constraints lack careful characterization.
* **Deliverables**: A fully reproducible **evaluation harness** (ir\_datasets + Pyserini/Lucene + sqlite‑vec or Faiss), per‑dataset **Pareto plots**, and a **thesis‑ready** write‑up with statistically validated findings.

---

## Corpora and query sets (small but representative, with gold labels)

Choose **3 BEIR datasets** spanning domains and difficulty, plus a tiny stress‑test set:

* **SciFact (claims → evidence)**: specialized, knowledge‑intensive; small and CPU‑friendly. Use BEIR split. ([arXiv][1], [ACL Anthology][2])
* **FiQA‑2018 (financial Q\&A)**: noisy, user‑style queries; medium difficulty. ([andrefreitas.org][3], [Google Sites][4])
* **TREC‑COVID (CORD‑19)**: long documents; tests chunking and long‑context effects. Use the BEIR/TREC‑COVID subset via ir\_datasets. ([arXiv][5], [ir-datasets.com][6])
* *(Optional micro set)*: **NQ‑lite** (subsampled BEIR NQ) if you want a general‑domain baseline; keep it small to fit memory/CPU. (BEIR provides NQ; see BEIR.) ([ir-datasets.com][7])

**Why these**: They are standard in **BEIR** (strong comparability and metrics), small enough to embed and index locally, and they stress different properties: domain specificity (SciFact), noisy/question‑like language (FiQA), and **long documents** that expose **chunking & context placement** problems (TREC‑COVID). ([ir-datasets.com][7])

---

## Query sets (how we’ll generate and audit them)

Use the **official BEIR queries/qrels** as the **primary** set (for NDCG\@10, Recall\@k, MAP). Then add **two controlled augmentations** per dataset (to test generalization without humans):

1. **LLM‑based expansions**:

   * **HyDE** (generate a hypothetical answer/document, embed it, and retrieve). Strong zero‑shot retrieval improvements, but watch for leakage. ([arXiv][8], [ACL Anthology][9])
   * **query2doc** (few‑shot prompt LLM to synthesize a pseudo‑doc; expand original query). Shown to boost BM25 by **3–15%** on MS MARCO/TREC DL. ([ACL Anthology][10])
   * **Risk control**: add **Corpus‑Steered Query Expansion (CSQE)**—use only sentences extracted from **already retrieved** in‑corpus texts, then expand; effective and mitigates leakage. ([ACL Anthology][11])
   * **Motivation**: a recent study questions whether HyDE gains partly come from **pretrain knowledge leakage**—CSQE gives you a cleaner baseline for analysis. ([arXiv][12])

2. **Perturbed queries** for robustness: synonym swaps, entity masking, and slight misspellings to evaluate **lexical vs. semantic** robustness (no new labels required; you still score with qrels).

*(All expansions run locally with your LLM; caps on token budgets keep cost low.)*

---

## Metrics and statistics (what and how we’ll measure)

* **Retrieval**: NDCG\@10 (primary), Recall\@k (k∈{1,5,20}), MRR\@10 when applicable. Use BEIR conventions for comparability. ([ir-datasets.com][7])
* **End‑to‑end RAG (optional)**: Exact Match / F1 only on datasets with ground truth answers (e.g., NQ‑lite), but this project is **retrieval‑centric**.
* **Latency & resources**: per‑query retrieval time (sparse, dense, fusion, rerank), embedding/build time, memory footprint (MB) for index & model.
* **Significance tests**: **paired t‑test** and **paired bootstrap** across topics/queries; both are widely endorsed in IR for robustness. Report effect sizes and confidence intervals. ([CIIR][13], [J-STAGE][14])

---

## Your modular system: what to plug where

* **Sparse**: Pyserini/Anserini (Lucene BM25; optional SPLADE if you want learned sparse). ([Peilin Yang][15], [Cheriton School of Computer Science][16])
* **Dense**: E5‑base‑v2 and BGE‑base‑en‑v1.5 (both strong and efficient); optional BGE‑M3 (multi‑functionality, multilingual, multi‑vector). ([arXiv][17], [Hugging Face][18])
* **Vector store**: **sqlite‑vec** (pure C SQLite extension; great portability and small footprint) or **Faiss** (more mature ANN/PQ options). Use one consistently per experiment series. ([Alex Garcia][19], [faiss.ai][20])
* **Rerankers**: **ms‑marco‑MiniLM‑L6‑v2** (6‑layer cross‑encoder) and **BGE‑reranker‑v2‑m3** (multilingual). Both run acceptably on CPU for **top‑50** candidates. ([Hugging Face][21])

---

## Six core experiments (grounded in literature, tuned for your Mac)

### E1 — Chunking under tight memory/CPU: fixed vs. semantic vs. **late chunking**

* **Why**: Long contexts often **hurt** retrieval‑augmented QA when relevant spans sit in the **middle**; chunking and context ordering matter. “Lost in the Middle” documents strong position effects and long‑context degradation. **Late chunking** (encode long context, then split) is promising but compute‑heavier. ([MIT Press Direct][22], [arXiv][23], [Open Review][24])
* **Design**:

  * Chunk sizes: **128, 256, 512** tokens; overlaps: **0, 20, 50**.
  * Strategies: (a) plain fixed, (b) **semantic** (split on headings/sentences), (c) **late chunking** (encode doc once with a long‑context embedder, then derive sub‑chunk vectors). ([Open Review][24])
  * Datasets: SciFact, TREC‑COVID, FiQA.
* **Hypotheses**:
  H1: **Semantic** and **late chunking** improve NDCG\@10 on **long docs** (TREC‑COVID), especially at **smaller k**.
  H2: **Smaller chunks (128–256)** win on SciFact/FiQA (claims, short passages).
* **Outputs**: Per‑dataset **NDCG\@10 vs. latency** curves; position‑sensitivity analysis by shuffling concatenated retrieved chunks (middle vs. edges) to probe the “lost in the middle” effect. ([MIT Press Direct][22])

### E2 — Embedding backbones & vectorization: E5 vs. BGE, dimensions & quantization

* **Why**: **MTEB** shows no universal winner; behavior varies by task. BGE and E5 are strong open baselines with instruction‑tuned variants; understanding their **speed/quality** under CPU/MPS is valuable. ([arXiv][25], [Hugging Face][26])
* **Design**:

  * Compare **E5‑base‑v2** vs. **BGE‑base‑en‑v1.5**; optionally **BGE‑M3**.
  * Dimensionality & storage: test **float32 vs. float16/int8** (if supported by your store), and **Matryoshka/truncation** if available. ([Hugging Face][27])
  * Keep chunking fixed at best setting from E1.
* **Hypotheses**:
  H3: BGE and E5 will **trade leads** across datasets (domain‑specific variance per MTEB). H4: **Half‑precision/quantized** vectors slightly reduce NDCG but give **substantial memory/latency** wins important for a 16 GB machine. ([arXiv][25])
* **Outputs**: Pareto plots (**NDCG\@10 vs. MB and ms/query**) and a recommended **default**.

### E3 — Hybrid retrieval: BM25 ⊕ dense (weighted sum vs. **RRF** vs. **Dynamic‑α**)

* **Why**: Dense and sparse are **complementary**; simple **interpolation** or **RRF** often outperforms either alone. Learning or adapting the **mix per query** (Dynamic‑α) is an open lever. ([ACL Anthology][28])
* **Design**:

  * Fusion: (a) **weighted sum** of z‑scored scores with **α∈{0.2,0.5,0.8}**, (b) **Reciprocal Rank Fusion (RRF)**, (c) **DAT** (Dynamic Alpha Tuning) using a small LLM to compare top‑1 BM25 vs. top‑1 dense, then set α per query. ([cormack.uwaterloo.ca][29], [arXiv][30])
  * Candidate pool sizes: N∈{50,100}.
* **Hypotheses**:
  H5: **RRF** is a strong, **training‑free** baseline on all three datasets. H6: **DAT** wins on **mixed query types** (FiQA) but adds small latency; it may not beat RRF on homogeneous SciFact. ([cormack.uwaterloo.ca][29], [arXiv][30])
* **Outputs**: **NDCG\@10** gains over best single retriever; **ablation** per query type (entity‑heavy vs. paraphrastic) to explain when hybrid helps most.

### E4 — Lightweight reranking under CPU constraints

* **Why**: Cross‑encoders (BERT/MonoT5) deliver large gains but can be slow; **MiniLM‑L6‑v2** and **BGE‑reranker‑v2‑m3** are practical small rerankers. Classic BERT reranking and multi‑stage pipelines (mono/duo) motivate this setup. ([arXiv][31])
* **Design**:

  * Inputs: top‑50 from the **best retriever/hybrid** in E3.
  * Rerankers: **MiniLM‑L6‑v2** and **BGE‑reranker‑v2‑m3**; batch size tuned for CPU/MPS. ([Hugging Face][21])
* **Hypotheses**:
  H7: Both rerankers **lift NDCG\@10 and Recall\@1**; MiniLM is **faster**; BGE‑v2‑m3 may be **more robust** cross‑domain.
* **Outputs**: **ΔNDCG\@10 vs. added latency**; choose a recommended **k and model** for your machine.

### E5 — LLM‑aided query expansion: **HyDE**, **query2doc**, **CSQE (corpus‑steered)**

* **Why**: HyDE/query2doc improve zero‑shot/sparse retrieval; concerns remain about **LLM leakage**. CSQE constrains expansions to **in‑corpus** text and has shown strong results. ([arXiv][8], [ACL Anthology][10])
* **Design**:

  * Compare **no‑expansion** vs. **HyDE** vs. **query2doc** vs. **CSQE** over BM25, dense, and the best hybrid from E3.
  * Limit expansion to ≤256 tokens; cache expansions.
* **Hypotheses**:
  H8: **CSQE** yields most of HyDE’s gains **without leakage risk**; **query2doc** gives the **largest boost to BM25** on FiQA; gains diminish once strong hybrid+rerank are in place. ([ACL Anthology][10])
* **Outputs**: Gains per retrieval mode and dataset; a **risk‑aware** recommendation (default to **CSQE**, enable HyDE only when allowed).

### E6 — Context budget & top‑k selection (bridging retrieval and generation)

* **Why**: Even with good retrieval, throwing too many chunks at the LLM can **hurt** (“lost in the middle”). We need **k** that optimizes answer quality **and** latency. ([MIT Press Direct][22])
* **Design**:

  * For the best pipeline so far, vary **k∈{2,4,6,8,12}**, randomly permute retrieved chunk order vs. place **most relevant first**.
  * Optional needle‑in‑haystack micro‑tests to validate position sensitivity. ([ukgovernmentbeis.github.io][32], [SpringerLink][33])
* **Hypothesis**:
  H9: **k=4–6** with **descending‑relevance ordering** dominates; larger k increases latency and degrades EM/F1 on long‑doc queries.

---

## Optional add‑ons (if time permits)

* **Learned sparse vs. BM25**: SPLADE/uniCOIL first‑stage vs. dense and hybrid; shows whether learned sparse beats BM25 on your hardware. ([arXiv][34])
* **Multi‑vector retrieval** (ColBERT/ColBERTv2 late‑interaction) for an accuracy ceiling check on small subsets. ([People at EECS][35], [arXiv][36])

---

## Rigorous evaluation harness (reproducible & thesis‑ready)

1. **Datasets** via `ir_datasets` + **BEIR** splits; use official qrels and trec‑eval compatible metrics. ([ir-datasets.com][7])
2. **Sparse** with **Pyserini/Anserini** (BM25; optional SPLADE); **Dense** with **E5/BGE**; **Hybrid** with score interpolation and **RRF** (simple, state‑of‑the‑art fusion). ([Cheriton School of Computer Science][16], [cormack.uwaterloo.ca][29])
3. **Vector store**:

   * **sqlite‑vec** if you value **simplicity/portability** and small memory. ([Alex Garcia][19])
   * **Faiss** if you need **HNSW/PQ/OPQ** variants for ANN speed/memory. ([faiss.ai][20])
4. **Reranking** with **MiniLM‑L6‑v2** and **BGE‑reranker‑v2‑m3**; measure ΔNDCG and latency vs. no rerank. ([Hugging Face][21])
5. **LLM expansions**: HyDE, query2doc, and **CSQE** pipelines. ([arXiv][8], [ACL Anthology][10])
6. **Statistics**: paired t‑test and paired bootstrap across queries; report p‑values and effect sizes (CIs). ([CIIR][13], [J-STAGE][14])

---

## Concrete procedures (per experiment)

For each dataset **D ∈ {SciFact, FiQA, TREC‑COVID}**:

1. **Index build**

   * Apply chunking variant (E1).
   * Embed with chosen model (E2 settings).
   * Build **BM25** index (Pyserini) and **dense** index (sqlite‑vec or Faiss). ([Cheriton School of Computer Science][16], [Alex Garcia][19])

2. **Primary runs**

   * **BM25**, **Dense**, **Hybrid (RRF + weighted α)**. ([cormack.uwaterloo.ca][29])
   * Add **reranking** (MiniLM or BGE‑v2‑m3) on top‑50. ([Hugging Face][21])
   * Record metrics and **latency** (end‑to‑end per query: retrieve → rerank).

3. **Augmented runs**

   * Re‑run with **HyDE**, **query2doc**, **CSQE**. Control token budgets; cache expansions. ([arXiv][8], [ACL Anthology][10])

4. **Analysis**

   * **Significance**: paired t‑test + bootstrap vs. baseline BM25 (or best single). ([CIIR][13])
   * **Breakdowns**: by query length, lexical rarity, entity presence; and by doc length (TREC‑COVID).
   * **Compute budget**: plot **NDCG\@10 vs. latency** and **memory**; identify **Pareto frontier**.

5. **Context budget (E6)**

   * For the **best pipeline** on D, sweep **k** and chunk ordering; measure EM/F1 where possible and retrieval‑conditioned answerability.

---

## Feasibility on Mac mini M4, 16 GB

* Use **384–768‑dim** embeddings, float16 if supported, to keep resident vector memory ≤ a few GB even for tens of thousands of chunks (e.g., 100k × 384 × 2 bytes ≈ **76 MB** vectors + index overhead).
* Keep **candidate pool** to **N ≤ 100** for rerankers; batch size **8–16** for MiniLM‑L6‑v2; **8** for BGE‑v2‑m3 on CPU is usually fine. ([Hugging Face][21])
* Prefer **sqlite‑vec** if you want minimal dependencies; pick **Faiss** if you need ANN variants (HNSW/PQ). ([Alex Garcia][19], [faiss.ai][20])

---

## What fills the knowledge gaps?

* **Chunking**: You’ll provide the first careful **small‑hardware** comparison of **fixed vs. semantic vs. late chunking** under BEIR‑style evaluation and long‑doc settings (TREC‑COVID), directly tying outcomes to **long‑context limitations** (Lost‑in‑the‑Middle). ([MIT Press Direct][22], [Open Review][24])
* **Hybrid retrieval**: You’ll compare **training‑free RRF** vs. **Dynamic‑α** on a **local** system and quantify **when** dynamic fusion is worth the overhead, building on evidence that sparse and dense are complementary. ([ACL Anthology][28], [cormack.uwaterloo.ca][29], [arXiv][30])
* **Query expansion**: You’ll contrast **HyDE** and **query2doc** with **CSQE** as a leakage‑aware alternative, clarifying when expansion still brings incremental gains once **hybrid + rerank** are strong. ([arXiv][8], [ACL Anthology][10])
* **Light rerankers**: You’ll put **CPU‑friendly** cross‑encoders on equal footing and chart the **quality/latency** curve, grounded in classic reranking literature. ([arXiv][31])

---

## Benchmarks and sources (for your Related Work section)

* **BEIR benchmark** and evaluation protocol. ([ir-datasets.com][7])
* **Long‑context limits** (“Lost in the Middle”). ([MIT Press Direct][22], [ACL Anthology][37])
* **Chunking** (late chunking: recent ICLR submission). ([Open Review][24])
* **Hybrid retrieval**: sparse+dense hybrids improve results; RRF is a strong, simple fusion; Dynamic‑α proposes per‑query mixing. ([ACL Anthology][28], [cormack.uwaterloo.ca][29], [arXiv][30])
* **Reranking**: BERT/MonoT5 and multi‑stage ranking foundations; small **MiniLM** and **BGE‑v2‑m3** are practical CPU options. ([arXiv][31], [Cheriton School of Computer Science][38], [Hugging Face][21])
* **LLM‑based expansion**: **HyDE**, **query2doc**, **CSQE** (+ leakage concerns). ([arXiv][8], [ACL Anthology][10])
* **Embedding model families**: **E5** technical report; **BGE** series & M3. ([arXiv][17], [Hugging Face][18])
* **Tooling**: **Pyserini/Anserini** (BM25/SPLADE; HNSW integration), **sqlite‑vec**, **Faiss**. ([Cheriton School of Computer Science][16], [arXiv][39], [Alex Garcia][19], [faiss.ai][20])
* **Significance testing** in IR (paired t‑test, bootstrap). ([CIIR][13], [J-STAGE][14])

---

## Suggested thesis structure and timeline (performance track)

**Phase 1 (1–2 weeks)** — *Harness & baselines*

* Implement dataset loaders (ir\_datasets), BM25 (Pyserini), one dense model, and evaluation scripts; confirm metrics vs. BEIR baselines. ([Cheriton School of Computer Science][16])

**Phase 2 (2–3 weeks)** — *E1 + E2*

* Chunking sweeps (fixed/semantic/late) and embedding backbones + precision/quantization. Produce Pareto plots.

**Phase 3 (2–3 weeks)** — *E3 + E4*

* Hybrid fusion (α‑sweep, RRF, DAT) and lightweight reranking. Record ΔNDCG/latency.

**Phase 4 (1–2 weeks)** — *E5 + E6*

* Query expansion (HyDE/query2doc/CSQE) and context budget/top‑k experiments.

**Phase 5 (1–2 weeks)** — *Analysis & write‑up*

* Statistical testing, ablation summaries, and the **guidelines** section (what to use when on small hardware).

---

## On “suitable corpus” and “suitable query set”: importance

* **Yes—critical** for performance claims: Using BEIR‑standard corpora and qrels makes your results **comparable and trustworthy**; expansions/perturbations probe **robustness** beyond the base distribution. ([ir-datasets.com][7])
* The above selection balances **domain diversity** with **local feasibility** (SciFact/FiQA/TREC‑COVID), and the query plan lets you quantify **generalization** without human subjects.

---

### Final note on practicality

Everything here runs locally with reasonable constraints (embedding dims ≤768, top‑k ≤100, CPU reranking). The outcome will be a high‑quality **performance map** of RAG techniques—**chunking**, **embedding**, **hybrid fusion**, **reranking**, **expansion**, and **context budgeting**—for **consumer Apple Silicon**, with strong academic grounding and replicable evaluation.

If you want, I can generate a check‑list of exact CLI commands and YAML configs for each experiment (per dataset, per variant) using your current modular repo layout.

[1]: https://arxiv.org/abs/2004.14974?utm_source=chatgpt.com "Fact or Fiction: Verifying Scientific Claims"
[2]: https://aclanthology.org/2020.emnlp-main.609/?utm_source=chatgpt.com "Fact or Fiction: Verifying Scientific Claims - ACL Anthology"
[3]: https://andrefreitas.org/papers/www_fiqa_2018.pdf?utm_source=chatgpt.com "WWW'18 Open Challenge: Financial Opinion Mining and Question Answering"
[4]: https://sites.google.com/view/fiqa/home?utm_source=chatgpt.com "FiQA - 2018 - Google Sites"
[5]: https://arxiv.org/abs/2104.09632?utm_source=chatgpt.com "Searching for Scientific Evidence in a Pandemic: An Overview of TREC-COVID"
[6]: https://ir-datasets.com/cord19.html?utm_source=chatgpt.com "CORD-19 - ir_datasets"
[7]: https://ir-datasets.com/msmarco-passage-v2.html?utm_source=chatgpt.com "MSMARCO (passage, version 2) - ir_datasets"
[8]: https://arxiv.org/pdf/2212.10496?utm_source=chatgpt.com "Precise Zero-Shot Dense Retrieval without Relevance Labels - arXiv.org"
[9]: https://aclanthology.org/2023.acl-long.99/?utm_source=chatgpt.com "Precise Zero-Shot Dense Retrieval without Relevance Labels"
[10]: https://aclanthology.org/2023.emnlp-main.585.pdf?utm_source=chatgpt.com "Query2doc: Query Expansion with Large Language Models"
[11]: https://aclanthology.org/2024.eacl-short.34.pdf?utm_source=chatgpt.com "Corpus-Steered Query Expansion with Large Language Models"
[12]: https://arxiv.org/abs/2504.14175?utm_source=chatgpt.com "Hypothetical Documents or Knowledge Leakage? Rethinking LLM-based Query ..."
[13]: https://ciir-publications.cs.umass.edu/getpdf.php?id=744&utm_source=chatgpt.com "A Comparison of Statistical Significance Tests for Information ..."
[14]: https://www.jstage.jst.go.jp/article/ipsjdc/3/0/3_0_625/_pdf?utm_source=chatgpt.com "48_14_2-DC.dvi - J-STAGE"
[15]: https://peilin-yang.github.io/files/pub/Yang_etal_SIGIR2017.pdf?utm_source=chatgpt.com "Anserini: Enabling the Use of Lucene for Information ... - Peilin Yang"
[16]: https://cs.uwaterloo.ca/~jimmylin/publications/Lin_etal_SIGIR2021_Pyserini.pdf?utm_source=chatgpt.com "Pyserini: A Python Toolkit for Reproducible Information Retrieval ..."
[17]: https://arxiv.org/pdf/2402.05672?utm_source=chatgpt.com "arXiv:2402.05672v1 [cs.CL] 8 Feb 2024"
[18]: https://huggingface.co/BAAI/bge-base-en?utm_source=chatgpt.com "BAAI/bge-base-en · Hugging Face"
[19]: https://alexgarcia.xyz/blog/2024/sqlite-vec-stable-release/index.html?utm_source=chatgpt.com "Introducing sqlite-vec v0.1.0: a vector search SQLite extension that ..."
[20]: https://faiss.ai/index.html?utm_source=chatgpt.com "Welcome to Faiss Documentation"
[21]: https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2?utm_source=chatgpt.com "cross-encoder/ms-marco-MiniLM-L6-v2 · Hugging Face"
[22]: https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00638/119630/Lost-in-the-Middle-How-Language-Models-Use-Long?utm_source=chatgpt.com "Lost in the Middle: How Language Models Use Long Contexts"
[23]: https://arxiv.org/abs/2307.03172v3?utm_source=chatgpt.com "Lost in the Middle: How Language Models Use Long Contexts"
[24]: https://openreview.net/forum?id=74QmBTV0Zf&utm_source=chatgpt.com "Late Chunking: Contextual Chunk Embeddings Using Long-Context..."
[25]: https://arxiv.org/pdf/2210.07316?utm_source=chatgpt.com "MTEB: Massive Text Embedding Benchmark - arXiv.org"
[26]: https://huggingface.co/spaces/mteb/leaderboard?utm_source=chatgpt.com "MTEB Leaderboard - a Hugging Face Space by mteb"
[27]: https://huggingface.co/BAAI/bge-m3?utm_source=chatgpt.com "BAAI/bge-m3 · Hugging Face"
[28]: https://aclanthology.org/2021.tacl-1.20.pdf?utm_source=chatgpt.com "Sparse, Dense, and Attentional Representations for Text Retrieval"
[29]: https://cormack.uwaterloo.ca/cormacksigir09-rrf.pdf?utm_source=chatgpt.com "Reciprocal Rank Fusion outperforms Condorcet and individual Rank ..."
[30]: https://arxiv.org/pdf/2503.23013?utm_source=chatgpt.com "DAT: Dynamic Alpha Tuning for Hybrid Retrieval in Retrieval-Augmented ..."
[31]: https://arxiv.org/pdf/1901.04085?utm_source=chatgpt.com "PASSAGE RE RANKING WITH BERT - arXiv.org"
[32]: https://ukgovernmentbeis.github.io/inspect_evals/evals/reasoning/niah/?utm_source=chatgpt.com "Needle in a Haystack (NIAH): In-Context Retrieval Benchmark for Long ..."
[33]: https://link.springer.com/chapter/10.1007/978-3-031-96196-0_19?utm_source=chatgpt.com "Needle-in-the-Haystack Testing LLMs with a Complex Reasoning Task"
[34]: https://arxiv.org/pdf/2107.05720?utm_source=chatgpt.com "SPLADE: Sparse Lexical and Expansion Model for First Stage Ranking"
[35]: https://people.eecs.berkeley.edu/~matei/papers/2020/sigir_colbert.pdf?utm_source=chatgpt.com "ColBERT: Efficient and Effective Passage Search via Contextualized Late ..."
[36]: https://arxiv.org/pdf/2112.01488?utm_source=chatgpt.com "ColBERTv2: Effective and Efﬁcient Retrieval via Lightweight Late Interacti"
[37]: https://aclanthology.org/2024.tacl-1.9/?utm_source=chatgpt.com "Lost in the Middle: How Language Models Use Long Contexts"
[38]: https://cs.uwaterloo.ca/~jimmylin/publications/Nogueira_etal_FindingsEMNLP2020.pdf?utm_source=chatgpt.com "Document Ranking with a Pretrained Sequence-to-Sequence Model"
[39]: https://arxiv.org/abs/2304.12139?utm_source=chatgpt.com "Anserini Gets Dense Retrieval: Integration of Lucene's HNSW Indexes"
