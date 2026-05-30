"""
Step 0 — Take ownership of the curriculum index.   (run once)

Two independent phases:

  --bm25     Rebuild the BM25 index from the existing OCR'd chunk texts using our
             Arabic-aware tokenizer (arabic_norm.tokenize). Fast, no downloads.
             Writes qdrant_storage/bm25_v2.pkl.

  --dense    Probe which embedding model produced the stored 1024-d vectors:
               • embed chunks[i]['text'] with BAAI/bge-m3 (then multilingual-e5-large)
               • compare cosine against the stored Qdrant vector for point i
             If a model matches (cosine ~1.0) → reuse the 'curriculum' collection as-is.
             Otherwise → re-embed every chunk text with bge-m3 into 'curriculum_v2'
             (reuses OCR text, no re-OCR). Downloads the model (~2 GB) on first run.

Default (no flag): run both.

Outputs:
  qdrant_storage/bm25_v2.pkl       {"bm25": BM25Okapi, "chunks": [...]}
  qdrant_storage/index_config.json {"embed_model","collection","dense_enabled","dim"}
  qdrant_storage/bm25.pkl.bak      one-time backup of the original

Run:
  python tools/build_index.py            # both
  python tools/build_index.py --bm25     # bm25 only (fast)
  python tools/build_index.py --dense    # dense probe / re-embed only
"""

import os
import sys
import json
import time
import shutil
import pickle
import argparse

# make the project root importable when run as `python tools/build_index.py`
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from arabic_norm import tokenize   # noqa: E402

QDRANT_PATH   = os.path.join(ROOT, "qdrant_storage")
ORIG_PKL      = os.path.join(QDRANT_PATH, "bm25.pkl")
NEW_PKL       = os.path.join(QDRANT_PATH, "bm25_v2.pkl")
BACKUP_PKL    = os.path.join(QDRANT_PATH, "bm25.pkl.bak")
CONFIG_PATH   = os.path.join(QDRANT_PATH, "index_config.json")

ORIG_COLLECTION = "curriculum"
NEW_COLLECTION  = "curriculum_v2"
VEC_DIM         = 1024

# We standardize on bge-m3 (the model we control). We still probe it against the
# stored vectors: if they happen to match we reuse them, otherwise we re-embed.
# (We do NOT probe e5 — it triggers a slow 2 GB download for no benefit.)
PROBE_MODELS = [
    ("BAAI/bge-m3", ""),
]
COSINE_MATCH = 0.92   # cosine above this ⇒ same model produced the stored vectors


# ── helpers ──────────────────────────────────────────────────────────────────

def _load_chunks() -> list[dict]:
    with open(ORIG_PKL, "rb") as f:
        obj = pickle.load(f)
    chunks = obj["chunks"]
    print(f"[build] loaded {len(chunks)} chunks from bm25.pkl")
    return chunks


def _read_config() -> dict:
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {}


def _write_config(**updates):
    cfg = _read_config()
    cfg.update(updates)
    cfg["built_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    print(f"[build] wrote index_config.json → {cfg}")


def _cosine(a, b) -> float:
    import numpy as np
    a = np.asarray(a, dtype="float64")
    b = np.asarray(b, dtype="float64")
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1.0
    return float(a @ b / denom)


# ── Phase: BM25 rebuild ──────────────────────────────────────────────────────

def rebuild_bm25(chunks: list[dict]):
    from rank_bm25 import BM25Okapi

    if not os.path.exists(BACKUP_PKL):
        shutil.copy2(ORIG_PKL, BACKUP_PKL)
        print(f"[bm25] backed up original → {os.path.basename(BACKUP_PKL)}")

    print("[bm25] tokenizing corpus with Arabic-aware tokenizer...")
    t0 = time.time()
    tokenized = [tokenize(c["text"]) for c in chunks]
    empties = sum(1 for d in tokenized if not d)
    print(f"[bm25] tokenized {len(tokenized)} docs in {time.time()-t0:.1f}s "
          f"({empties} empty)")

    print("[bm25] fitting BM25Okapi...")
    bm25 = BM25Okapi(tokenized)
    with open(NEW_PKL, "wb") as f:
        pickle.dump({"bm25": bm25, "chunks": chunks}, f)
    print(f"[bm25] saved → {os.path.basename(NEW_PKL)}  "
          f"(corpus={bm25.corpus_size}, avgdl={bm25.avgdl:.1f})")
    _write_config(bm25_file="bm25_v2.pkl")


# ── Phase: dense probe / re-embed ────────────────────────────────────────────

def _get_stored_vectors(client, ids):
    recs = client.retrieve(collection_name=ORIG_COLLECTION, ids=ids, with_vectors=True)
    by_id = {r.id: r.vector for r in recs}
    return [by_id.get(i) for i in ids]


def probe_or_reembed(chunks: list[dict]):
    from qdrant_client import QdrantClient
    from sentence_transformers import SentenceTransformer
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    client = QdrantClient(path=QDRANT_PATH)

    probe_ids = [0, 500, 5000, 11000]
    probe_ids = [i for i in probe_ids if i < len(chunks)]
    stored = _get_stored_vectors(client, probe_ids)
    if any(v is None for v in stored):
        print("[dense] WARN: could not read some stored vectors; ids may differ.")
    print(f"[dense] probing with sample ids {probe_ids} on {device}")

    for model_name, prefix in PROBE_MODELS:
        print(f"[dense] trying {model_name} ...")
        try:
            model = SentenceTransformer(model_name, device=device)
        except Exception as e:
            print(f"[dense]   load failed: {e}")
            continue
        texts = [prefix + chunks[i]["text"] for i in probe_ids]
        embs  = model.encode(texts, normalize_embeddings=True)
        sims  = [_cosine(embs[k], stored[k]) for k in range(len(probe_ids))
                 if stored[k] is not None]
        avg = sum(sims) / len(sims) if sims else 0.0
        print(f"[dense]   cosine vs stored: {[round(s,3) for s in sims]}  (avg {avg:.3f})")
        if avg >= COSINE_MATCH:
            print(f"[dense] MATCH → reuse '{ORIG_COLLECTION}' with {model_name}")
            client.close()
            _write_config(embed_model=model_name, embed_prefix=prefix,
                          collection=ORIG_COLLECTION, dense_enabled=True, dim=VEC_DIM)
            return

    # No match → re-embed with bge-m3 into a fresh collection
    model_name, prefix = PROBE_MODELS[0]
    print(f"[dense] no match — re-embedding all chunks with {model_name} → '{NEW_COLLECTION}'")
    model = SentenceTransformer(model_name, device=device)
    _reembed(client, model, prefix, chunks)
    client.close()
    _write_config(embed_model=model_name, embed_prefix=prefix,
                  collection=NEW_COLLECTION, dense_enabled=True, dim=VEC_DIM)


def _reembed(client, model, prefix, chunks):
    from qdrant_client.models import Distance, VectorParams, PointStruct

    model.max_seq_length = 512    # our chunks are short — cap padding for speed/VRAM

    if client.collection_exists(NEW_COLLECTION):
        client.delete_collection(NEW_COLLECTION)
    client.create_collection(
        collection_name=NEW_COLLECTION,
        vectors_config=VectorParams(size=VEC_DIM, distance=Distance.COSINE),
    )

    batch = 32
    t0 = time.time()
    for start in range(0, len(chunks), batch):
        part = chunks[start:start + batch]
        texts = [prefix + c["text"] for c in part]
        embs = model.encode(texts, normalize_embeddings=True, batch_size=batch)
        points = [
            PointStruct(
                id=start + j,
                vector=embs[j].tolist(),
                payload={
                    "book":     c["book"],
                    "subject":  c["subject"],
                    "grade":    c["grade"],
                    "page_num": c["page_num"],
                },
            )
            for j, c in enumerate(part)
        ]
        client.upsert(collection_name=NEW_COLLECTION, points=points)
        if start % (batch * 20) == 0:
            print(f"[dense]   embedded {start + len(part)}/{len(chunks)} "
                  f"({time.time()-t0:.0f}s)")
    print(f"[dense] re-embedded {len(chunks)} chunks in {time.time()-t0:.0f}s")


# ── entry ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bm25",  action="store_true", help="rebuild BM25 only")
    ap.add_argument("--dense", action="store_true", help="dense probe / re-embed only")
    args = ap.parse_args()

    do_bm25  = args.bm25  or not (args.bm25 or args.dense)
    do_dense = args.dense or not (args.bm25 or args.dense)

    sys.stdout.reconfigure(line_buffering=True)   # show progress when piped to a file
    chunks = _load_chunks()
    if do_bm25:
        rebuild_bm25(chunks)
    if do_dense:
        probe_or_reembed(chunks)
    print("[build] done.")


if __name__ == "__main__":
    main()
