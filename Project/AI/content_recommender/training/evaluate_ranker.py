"""
Evaluate the fine-tuned ranker vs the baseline on MIRACL-Arabic.
Reports MRR@10 and Recall@100 — standard IR metrics for your senior project report.

Run AFTER train_ranker.py finishes.
"""

from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

BASE_MODEL    = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
FINETUNED_DIR = Path(__file__).parent / "checkpoints" / "edu_ranker_ar"
TOP_K         = 100
NUM_QUERIES   = 200          # subsample for speed; full MIRACL = ~2,400
NUM_PASSAGES  = 50_000       # subsample passage pool


def encode(model, texts, batch_size=64):
    return model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )


def mrr_at_k(ranked_relevances, k):
    out = []
    for r in ranked_relevances:
        rank = next((i + 1 for i, rel in enumerate(r[:k]) if rel > 0), None)
        out.append(1.0 / rank if rank else 0.0)
    return float(np.mean(out))


def recall_at_k(ranked_relevances, k, total_relevant_per_query):
    out = []
    for r, total in zip(ranked_relevances, total_relevant_per_query):
        if total == 0:
            continue
        hits = sum(1 for rel in r[:k] if rel > 0)
        out.append(hits / total)
    return float(np.mean(out)) if out else 0.0


def evaluate(model_name_or_path, queries_data, passages, qrels):
    print(f"\n=== Evaluating {model_name_or_path} ===")
    model = SentenceTransformer(str(model_name_or_path))

    q_texts = [q["query"] for q in queries_data]
    q_ids   = [q["id"]    for q in queries_data]

    q_emb = encode(model, q_texts)
    p_emb = encode(model, [p["text"] for p in passages])
    p_ids = [p["id"] for p in passages]
    p_id_index = {pid: i for i, pid in enumerate(p_ids)}

    # Cosine sim (vectors are normalized → dot product)
    sims = q_emb @ p_emb.T

    ranked_relevances = []
    total_relevant_per_query = []
    for qi, qid in enumerate(q_ids):
        relevant_pids = qrels.get(qid, set())
        total_relevant_per_query.append(len(relevant_pids))

        top_indices = np.argsort(-sims[qi])[:TOP_K]
        rels = [1 if p_ids[i] in relevant_pids else 0 for i in top_indices]
        ranked_relevances.append(rels)

    mrr10  = mrr_at_k(ranked_relevances, 10)
    rec100 = recall_at_k(ranked_relevances, 100, total_relevant_per_query)
    print(f"  MRR@10      = {mrr10:.4f}")
    print(f"  Recall@100  = {rec100:.4f}")
    return {"mrr@10": mrr10, "recall@100": rec100}


def load_miracl():
    print("Loading MIRACL-Arabic...")
    qs   = load_dataset("miracl/miracl",        "ar", split="dev",   trust_remote_code=True)
    corp = load_dataset("miracl/miracl-corpus", "ar", split="train", trust_remote_code=True, streaming=True)

    queries: list[dict] = []
    qrels: dict[str, set] = {}
    relevant_pids: set[str] = set()

    for row in qs.select(range(min(NUM_QUERIES, len(qs)))):
        qid = row["query_id"]
        queries.append({"id": qid, "query": row["query"]})
        pos = {p["docid"] for p in row.get("positive_passages") or []}
        qrels[qid] = pos
        relevant_pids.update(pos)

    passages: list[dict] = []
    seen: set[str] = set()
    # Always include the relevant passages, then top up with random ones
    for row in corp:
        pid = row["docid"]
        if pid in relevant_pids and pid not in seen:
            passages.append({"id": pid, "text": row["text"]})
            seen.add(pid)
        if len(passages) >= len(relevant_pids):
            break

    for row in corp:
        pid = row["docid"]
        if pid in seen:
            continue
        passages.append({"id": pid, "text": row["text"]})
        seen.add(pid)
        if len(passages) >= NUM_PASSAGES:
            break

    print(f"  Queries: {len(queries)} | Passages: {len(passages)}")
    return queries, passages, qrels


def main():
    print(f"CUDA available: {torch.cuda.is_available()}")
    queries, passages, qrels = load_miracl()

    baseline   = evaluate(BASE_MODEL,    queries, passages, qrels)
    finetuned  = evaluate(FINETUNED_DIR, queries, passages, qrels) \
                 if FINETUNED_DIR.exists() else None

    print("\n" + "=" * 50)
    print("RESULTS SUMMARY")
    print("=" * 50)
    print(f"Baseline   MRR@10={baseline['mrr@10']:.4f}  Recall@100={baseline['recall@100']:.4f}")
    if finetuned:
        d_mrr = (finetuned['mrr@10'] - baseline['mrr@10']) / max(baseline['mrr@10'], 1e-6) * 100
        d_rec = (finetuned['recall@100'] - baseline['recall@100']) / max(baseline['recall@100'], 1e-6) * 100
        print(f"Fine-tuned MRR@10={finetuned['mrr@10']:.4f}  Recall@100={finetuned['recall@100']:.4f}")
        print(f"           Δ MRR@10  = {d_mrr:+.1f}%")
        print(f"           Δ Recall  = {d_rec:+.1f}%")
    else:
        print("Fine-tuned model not found — run train_ranker.py first.")


if __name__ == "__main__":
    main()
