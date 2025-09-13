#!/usr/bin/env bash
set -euo pipefail

# Finalize an experiment into experiments/finished/<slug>
# Copies curated artifacts, writes MANIFEST.yaml, and updates the experiments catalog.

SLUG=${1:-beir_v2_2_reranking}
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

DEST="experiments/finished/${SLUG}"
RESULTS_SRC="experiments/reranking/results"
REPORT_SRC_MD="experiments/reranking/FINAL_REPORT_v2_2_BEIR.md"
PR_NOTE_SRC="experiments/reranking/pr_summary_v2_2.txt"
PAPER_SRC_DIR="papers/beir_v2_2_reranking"

mkdir -p "$DEST/results" "$DEST/reports/paper" "$DEST/scripts" "$DEST/env" "$DEST/logs"

echo "Copying metric JSONs..."
find "$RESULTS_SRC" -maxdepth 1 -name "*.metrics.json" -type f -print -exec cp {} "$DEST/results/" \;
echo "Copying summary files..."
for f in summary.json summary.md; do
  [ -f "$RESULTS_SRC/$f" ] && cp "$RESULTS_SRC/$f" "$DEST/results/" || true
done

echo "Copying reports..."
[ -f "$REPORT_SRC_MD" ] && cp "$REPORT_SRC_MD" "$DEST/reports/" || true
[ -f "$PR_NOTE_SRC" ] && cp "$PR_NOTE_SRC" "$DEST/reports/" || true

echo "Copying paper artifacts..."
if [ -d "$PAPER_SRC_DIR" ]; then
  cp -R "$PAPER_SRC_DIR/"* "$DEST/reports/paper/"
fi

echo "Copying key scripts..."
cp experiments/reranking/run_reranking_experiment.py "$DEST/scripts/"
cp experiments/reranking/evaluate_reranking.py "$DEST/scripts/"
cp experiments/reranking/summarize_results.py "$DEST/scripts/" 2>/dev/null || true
cp experiments/reranking/generate_final_report.py "$DEST/scripts/" 2>/dev/null || true

echo "Exporting environment... (best-effort)"
if command -v conda >/dev/null 2>&1; then
  source ~/miniforge3/etc/profile.d/conda.sh >/dev/null 2>&1 || true
  conda env export -n rag_env > "$DEST/env/conda_environment.yml" 2>/dev/null || true
fi
pip freeze > "$DEST/env/pip_freeze.txt" 2>/dev/null || true

echo "Saving logs (tail)..."
for log in logs/fiqa_hybrid_sweeps_*.log logs/bge_confirms_*.log logs/finalize_reranking_*.log; do
  for f in $log; do
    [ -f "$f" ] || continue
    bn=$(basename "$f")
    tail -n 400 "$f" > "$DEST/logs/${bn%.log}.tail.log" || true
  done
done

echo "Writing MANIFEST.yaml..."
python - << 'PY'
import json, os, subprocess
from pathlib import Path
from datetime import datetime

slug = os.environ.get('SLUG', 'beir_v2_2_reranking')
dest = Path(f'experiments/finished/{slug}')
results = dest / 'results'
summary_json = results / 'summary.json'

def sh(cmd):
    return subprocess.check_output(cmd, shell=True, text=True).strip()

def load_rows():
    if not summary_json.exists():
        return []
    return json.loads(summary_json.read_text())

rows = load_rows()

def best_by_ndcg(dataset):
    subset = [r for r in rows if r.get('dataset') == dataset]
    return max(subset, key=lambda r: float(r.get('ndcg_at_10') or 0), default=None)

def baseline(dataset):
    subset = [r for r in rows if r.get('dataset') == dataset and 'baseline' in r.get('run_name','')]
    if not subset:
        subset = [r for r in rows if r.get('dataset') == dataset and not r.get('reranker_model')]
    return subset[0] if subset else None

fiqa_best = best_by_ndcg('fiqa')
scifact_best = best_by_ndcg('scifact')
fiqa_base = baseline('fiqa')
scifact_base = baseline('scifact')

def uniq(rows, key):
    vals = []
    for r in rows:
        v = r.get(key)
        if v is not None and v not in vals:
            vals.append(v)
    return vals

rerankers = uniq(rows, 'reranker_model')
alphas = uniq(rows, 'alpha')
cms = uniq(rows, 'candidate_multiplier')
topks = uniq(rows, 'rerank_topk')

commit = sh('git rev-parse HEAD')
branch = sh('git branch --show-current')
remote = sh('git config --get remote.origin.url') or ''
try:
    mb = sh('git merge-base origin/main HEAD')
except subprocess.CalledProcessError:
    mb = ''

manifest = {
    'title': 'BEIR v2.2 Reranking',
    'slug': slug,
    'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'repo': {
        'remote': remote,
        'branch': branch,
        'commit': commit,
        'merge_base_origin_main': mb,
    },
    'data': {
        'db_path': 'data/rag_vectors.db',
        'collections': ['fiqa_technical', 'scifact_scientific'],
        'queries': {
            'fiqa': 'test_data/beir_fiqa_queries_test_only.json',
            'scifact': 'test_data/beir_scifact_queries_test_only.json'
        }
    },
    'evaluation': {
        'k': 10,
        'rerank_topk_values': topks,
        'metrics': ['ndcg@10', 'recall@10']
    },
    'first_stage': {
        'methods': ['vector', 'keyword', 'hybrid'],
        'alphas': alphas,
        'candidate_multipliers': cms
    },
    'rerankers': rerankers,
    'results': {
        'fiqa': {
            'baseline': fiqa_base,
            'best': fiqa_best,
        },
        'scifact': {
            'baseline': scifact_base,
            'best': scifact_best,
        }
    }
}

(dest / 'MANIFEST.yaml').write_text(
    '\n'.join([
        'title: ' + manifest['title'],
        'slug: ' + manifest['slug'],
        'created_at: ' + manifest['created_at'],
        'repo:',
        f"  remote: '{manifest['repo']['remote']}'",
        f"  branch: '{manifest['repo']['branch']}'",
        f"  commit: '{manifest['repo']['commit']}'",
        f"  merge_base_origin_main: '{manifest['repo']['merge_base_origin_main']}'",
        'data:',
        "  db_path: 'data/rag_vectors.db'",
        "  collections: ['fiqa_technical','scifact_scientific']",
        "  queries:",
        "    fiqa: 'test_data/beir_fiqa_queries_test_only.json'",
        "    scifact: 'test_data/beir_scifact_queries_test_only.json'",
        'evaluation:',
        '  k: 10',
        f"  rerank_topk_values: {topks}",
        "  metrics: ['ndcg@10','recall@10']",
        'first_stage:',
        f"  methods: ['vector','keyword','hybrid']",
        f"  alphas: {alphas}",
        f"  candidate_multipliers: {cms}",
        'rerankers: ' + json.dumps(rerankers),
        'results:',
        '  fiqa:',
        '    baseline: ' + json.dumps(fiqa_base or {}),
        '    best: ' + json.dumps(fiqa_best or {}),
        '  scifact:',
        '    baseline: ' + json.dumps(scifact_base or {}),
        '    best: ' + json.dumps(scifact_best or {}),
        ''
    ])
)
print('Wrote MANIFEST.yaml')
PY

echo "Updating experiments catalog..."
mkdir -p experiments
if [ ! -f experiments/catalog.json ]; then
  echo "[]" > experiments/catalog.json
fi

python - << 'PY'
import json, os
from pathlib import Path

slug = os.environ.get('SLUG','beir_v2_2_reranking')
catalog = Path('experiments/catalog.json')
entries = json.loads(catalog.read_text()) if catalog.exists() else []
summary_md = f'experiments/finished/{slug}/results/summary.md'
report_md = f'experiments/finished/{slug}/reports/FINAL_REPORT_v2_2_BEIR.md'
manifest = f'experiments/finished/{slug}/MANIFEST.yaml'

# Avoid duplicate entries
entries = [e for e in entries if e.get('slug') != slug]
entries.append({
  'slug': slug,
  'title': 'BEIR v2.2 Reranking',
  'path': f'experiments/finished/{slug}',
  'report': report_md,
  'summary': summary_md,
  'manifest': manifest
})

catalog.write_text(json.dumps(entries, indent=2))
print('Catalog updated')
PY

if [ ! -f experiments/README.md ]; then
  cat > experiments/README.md << 'MD'
# Experiments Catalog

This directory tracks completed experiments and their curated artifacts.

- Index: `experiments/catalog.json`
- Finished experiments under: `experiments/finished/<slug>/`

Each finished experiment includes:
- `MANIFEST.yaml`: metadata (repo, data, parameters, best results)
- `results/`: metrics JSONs, summary tables
- `reports/`: final report, PR summary, paper files
- `scripts/`: key scripts used to run/evaluate
- `env/`: environment exports
- `logs/`: trimmed logs for provenance
MD
fi

echo "Finalize complete → $DEST"

