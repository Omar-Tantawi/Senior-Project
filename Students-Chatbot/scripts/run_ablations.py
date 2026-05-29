"""
Ablation runner for the Arabic Educational Chatbot.
Runs 4 ablations against eval_questions.json and writes ablation_results.json.

Ablations
---------
  baseline      -- full system (used as comparison baseline for all ablations)
  A2-1          -- no morphological normalization (CAMeL Tools bypassed at query time)
  A2-2          -- semantic-only retrieval (BM25 disabled, dense vectors only)
  A2-3          -- no similarity guardrail (threshold forced to 0.0)
  A2-4          -- MSA-only responses (Syrian colloquial layer removed from system prompt)

Metrics
-------
  Recall@5      -- % of questions where the source chunk (book + page) appears in top-5
                   Used for: baseline, A2-1, A2-2  (retrieval-only, no LLM calls)
  Bypass rate   -- % of off-curriculum questions that receive a real answer instead of
                   the out-of-scope reply.  Used for: A2-3 vs baseline (20 LLM calls)
  Dual-layer %  -- % of responses containing the Syrian-colloquial marker.
                   Used for: A2-4 vs baseline (60 LLM calls on a 30-question sample)

Usage
-----
    python scripts/run_ablations.py
    python scripts/run_ablations.py --eval eval_questions.json --out ablation_results.json
    python scripts/run_ablations.py --skip-llm   # retrieval ablations only (fastest)
"""
import sys
import json
import random
import argparse
import pickle
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

import ollama
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

from chatbot.config import (
    QDRANT_DIR, BM25_PATH, COLLECTION_NAME,
    EMBEDDING_MODEL, SIMILARITY_THRESHOLD,
    TOP_K_DENSE, TOP_K_BM25, TOP_K_FINAL, LLM_MODEL,
)
from chatbot.indexing.normalizer import normalize
from chatbot.llm.client import _build_context

# -- Constants -----------------------------------------------------------------

_RRF_K = 60

_NO_CONTEXT_REPLY = "هذا السؤال خارج المنهج الذي لديّ."

_SYSTEM_PROMPT_FULL = (
    "أنت مساعد تعليمي"
    " ذكي لطلاب المرحلة"
    " الثانوية في سوريا.\n"
    "مهمتك الإجابة على"
    " أسئلة الطلاب بناء"
    "ً على المناهج الدراسية"
    " السورية المقدمة فقط.\n\n"
    "قواعد صارمة:\n"
    "١. أجب دائمًا بطريقتين"
    " متتاليتين:\n"
    "   \U0001f5e3️ شرح بسيط:"
    " اشرح الفكرة بالعامية"
    " السورية.\n"
    "   \U0001f4d6 تعريف رسمي:"
    " قدم التعريف الأكاديمي"
    " بالعربية الفصحى.\n"
    "٢. للمسائل الرياضية"
    " والعلمية: اتبع أسلوب"
    " التفكير خطوة بخطوة.\n"
    "٣. لا تخرج عن المحتوى"
    " المقدم. إذا لم يكن"
    " الجواب في المحتوى،"
    " أجب: \"هذا السؤال خارج"
    " المنهج الذي لديّ.\"\n"
    "٤. لا تخترع أرقامًا"
    " أو معادلات غير موجودة.\n"
)

_SYSTEM_PROMPT_MSA = (
    "أنت مساعد تعليمي"
    " ذكي لطلاب المرحلة"
    " الثانوية في سوريا.\n"
    "مهمتك الإجابة على"
    " أسئلة الطلاب بناء"
    "ً على المناهج الدراسية"
    " السورية المقدمة فقط.\n\n"
    "قواعد صارمة:\n"
    "١. أجب دائمًا بالعربية"
    " الفصحى (MSA) فقط بأسلوب"
    " أكاديمي واضح.\n"
    "٢. للمسائل الرياضية"
    " والعلمية: اتبع أسلوب"
    " التفكير خطوة بخطوة.\n"
    "٣. لا تخرج عن المحتوى"
    " المقدم. إذا لم يكن"
    " الجواب في المحتوى،"
    " أجب: \"هذا السؤال خارج"
    " المنهج الذي لديّ.\"\n"
    "٤. لا تخترع أرقامًا"
    " أو معادلات غير موجودة.\n"
)

OFF_CURRICULUM_QUESTIONS = [
    "كيف أفتح حسابًا مصرفيًا؟",
    "ما هو سعر الدولار اليوم؟",
    "كيف أطبخ المنسف؟",
    "من هو مؤسس شركة آبل؟",
    "ما هي أفضل طريقة لتعلم لغة جديدة؟",
    "كيف أقدم طلب تأشيرة سفر؟",
    "ما هي أعراض نزلة البرد؟",
    "كيف أصلح إطار سيارة مثقوب؟",
    "ما هو أفضل هاتف ذكي في السوق؟",
    "كيف أنشئ حسابًا على إنستغرام؟",
]

# -- Shared resource loader ----------------------------------------------------

_model: SentenceTransformer | None = None
_client: QdrantClient | None = None
_bm25_data: dict | None = None


def _resources():
    global _model, _client, _bm25_data
    if _model is None:
        print("Loading embedding model...")
        _model = SentenceTransformer(EMBEDDING_MODEL, trust_remote_code=True)
    if _client is None:
        _client = QdrantClient(path=str(QDRANT_DIR))
    if _bm25_data is None:
        with open(BM25_PATH, "rb") as f:
            _bm25_data = pickle.load(f)
    return _model, _client, _bm25_data


# -- Parameterised retrieve ----------------------------------------------------

def retrieve_with_config(
    query: str,
    normalize_query: bool = True,
    use_bm25: bool = True,
    threshold: float = SIMILARITY_THRESHOLD,
) -> list[dict]:
    model, client, bm25_data = _resources()

    q = normalize(query) if normalize_query else query

    query_vec = model.encode(
        q, task="retrieval.query", normalize_embeddings=True
    ).tolist()

    dense_hits = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vec,
        limit=TOP_K_DENSE,
        with_payload=True,
        score_threshold=threshold,
    ).points

    # Guardrail: nothing cleared the threshold → off-curriculum, return nothing.
    # Skip this guard when threshold=0.0 (A2-3 ablation deliberately disables it).
    if not dense_hits and threshold > 0:
        return []

    rrf: dict[int, float] = {}
    for rank, hit in enumerate(dense_hits):
        rrf[hit.id] = rrf.get(hit.id, 0.0) + 1 / (_RRF_K + rank + 1)

    if use_bm25:
        bm25_scores = bm25_data["bm25"].get_scores(q.split())
        bm25_ranked = sorted(
            enumerate(bm25_scores), key=lambda x: x[1], reverse=True
        )[:TOP_K_BM25]
        for rank, (idx, _) in enumerate(bm25_ranked):
            rrf[idx] = rrf.get(idx, 0.0) + 1 / (_RRF_K + rank + 1)

    top_ids = sorted(rrf, key=lambda i: rrf[i], reverse=True)[:TOP_K_FINAL]

    results = []
    for doc_id in top_ids:
        hit = client.retrieve(COLLECTION_NAME, ids=[doc_id], with_payload=True)
        if not hit:
            continue
        payload = hit[0].payload
        dense_score = next((h.score for h in dense_hits if h.id == doc_id), None)
        if dense_score is None or dense_score >= threshold:
            results.append({**payload, "score": rrf[doc_id]})

    return results


# -- LLM call ------------------------------------------------------------------

def call_llm(question: str, chunks: list[dict], system_prompt: str) -> str:
    if not chunks:
        return _NO_CONTEXT_REPLY
    context = _build_context(chunks)
    user_message = f"المحتوى المرجعي:\n{context}\n\nسؤال الطالب: {question}"
    resp = ollama.chat(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_message},
        ],
        options={"temperature": 0.1},
    )
    return resp["message"]["content"]


# -- Metrics -------------------------------------------------------------------

def recall_at_k(chunks: list[dict], source_book: str, source_page: int, k: int = 5) -> bool:
    for c in chunks[:k]:
        if c.get("book") == source_book and abs((c.get("page_num") or 0) - source_page) <= 1:
            return True
    return False


def has_dual_layer(response: str) -> bool:
    return "شرح بسيط" in response


def is_blocked(response: str) -> bool:
    return _NO_CONTEXT_REPLY in response


# -- Ablation runners ----------------------------------------------------------

def run_retrieval_ablation(name: str, questions: list[dict], normalize_query: bool, use_bm25: bool) -> dict:
    hits = 0
    details = []
    for q in tqdm(questions, desc=f"  {name}"):
        chunks = retrieve_with_config(
            q["question"],
            normalize_query=normalize_query,
            use_bm25=use_bm25,
        )
        hit = recall_at_k(chunks, q.get("book", ""), q.get("page_num", 0))
        hits += int(hit)
        details.append({
            "id": q["id"], "subject": q["subject"],
            "hit": hit, "retrieved_books": [c.get("book") for c in chunks],
        })
    recall = hits / len(questions) if questions else 0.0
    return {"recall_at_5": round(recall, 4), "hits": hits, "total": len(questions), "details": details}


def run_guardrail_ablation(threshold: float) -> dict:
    bypassed = 0
    details = []
    for q_text in tqdm(OFF_CURRICULUM_QUESTIONS, desc=f"  guardrail (threshold={threshold})"):
        chunks = retrieve_with_config(q_text, threshold=threshold)
        response = call_llm(q_text, chunks, _SYSTEM_PROMPT_FULL)
        bypassed_flag = not is_blocked(response)
        bypassed += int(bypassed_flag)
        details.append({
            "question": q_text,
            "bypassed": bypassed_flag,
            "response_preview": response[:120],
        })
    bypass_rate = bypassed / len(OFF_CURRICULUM_QUESTIONS)
    return {"bypass_rate": round(bypass_rate, 4), "bypassed": bypassed,
            "total": len(OFF_CURRICULUM_QUESTIONS), "details": details}


def run_prompt_ablation(name: str, questions: list[dict], system_prompt: str) -> dict:
    compliant = 0
    details = []
    for q in tqdm(questions, desc=f"  {name}"):
        chunks = retrieve_with_config(q["question"])
        response = call_llm(q["question"], chunks, system_prompt)
        dual = has_dual_layer(response)
        compliant += int(dual)
        details.append({
            "id": q["id"], "subject": q["subject"],
            "dual_layer": dual, "response_preview": response[:200],
        })
    rate = compliant / len(questions) if questions else 0.0
    return {"dual_layer_rate": round(rate, 4), "compliant": compliant,
            "total": len(questions), "details": details}


# -- Main ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval",     default="eval_questions.json")
    parser.add_argument("--out",      default="ablation_results.json")
    parser.add_argument("--seed",     type=int, default=42)
    parser.add_argument("--sample",   type=int, default=30,
                        help="Questions for LLM-based ablations")
    parser.add_argument("--skip-llm", action="store_true",
                        help="Skip A2-3 and A2-4 (retrieval ablations only)")
    parser.add_argument("--guardrail-only", action="store_true",
                        help="Re-run only guardrail ablations, loading prior results for the rest")
    args = parser.parse_args()

    random.seed(args.seed)

    with open(args.eval, encoding="utf-8") as f:
        all_questions = json.load(f)

    # --guardrail-only: reload existing results, re-run only A2-3, save + print.
    if args.guardrail_only:
        out_path = Path(args.out)
        with open(out_path, encoding="utf-8") as f:
            results = json.load(f)

        print(f"\n[Baseline] Guardrail test -- {len(OFF_CURRICULUM_QUESTIONS)} off-curriculum questions")
        results["baseline_guardrail"] = run_guardrail_ablation(threshold=SIMILARITY_THRESHOLD)

        print(f"\n[A2-3] Guardrail disabled (threshold=0.0)")
        results["a2_3_no_guardrail"] = run_guardrail_ablation(threshold=0.0)

        results["summary"]["bypass_baseline"] = results["baseline_guardrail"]["bypass_rate"]
        results["summary"]["bypass_a2_3"]     = results["a2_3_no_guardrail"]["bypass_rate"]

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"Updated results saved -> {out_path}")

        b_g = results["baseline_guardrail"]["bypass_rate"]
        g3  = results["a2_3_no_guardrail"]["bypass_rate"]
        print(f"\nGuardrail bypass rate (out of {len(OFF_CURRICULUM_QUESTIONS)} off-curriculum Qs):")
        print(f"  Baseline (threshold=0.60)     : {b_g:.2%}")
        print(f"  A2-3 No guardrail (thresh=0)  : {g3:.2%}  (delta {g3-b_g:+.2%})")
        return

    by_subject = defaultdict(list)
    for q in all_questions:
        by_subject[q["subject"]].append(q)

    per_subject_sample = args.sample // 3
    llm_sample = []
    for subj, qs in by_subject.items():
        llm_sample.extend(random.sample(qs, min(per_subject_sample, len(qs))))

    results = {}

    print("\n[Baseline] Recall@5 -- hybrid retrieval + normalization")
    results["baseline_retrieval"] = run_retrieval_ablation(
        "baseline", all_questions, normalize_query=True, use_bm25=True
    )

    print("\n[A2-1] Recall@5 -- normalization disabled")
    results["a2_1_no_normalize"] = run_retrieval_ablation(
        "a2_1", all_questions, normalize_query=False, use_bm25=True
    )

    print("\n[A2-2] Recall@5 -- BM25 disabled (dense only)")
    results["a2_2_dense_only"] = run_retrieval_ablation(
        "a2_2", all_questions, normalize_query=True, use_bm25=False
    )

    if not args.skip_llm:
        print(f"\n[Baseline] Guardrail test -- {len(OFF_CURRICULUM_QUESTIONS)} off-curriculum questions")
        results["baseline_guardrail"] = run_guardrail_ablation(threshold=SIMILARITY_THRESHOLD)

        print(f"\n[A2-3] Guardrail disabled (threshold=0.0)")
        results["a2_3_no_guardrail"] = run_guardrail_ablation(threshold=0.0)

        print(f"\n[Baseline] Dual-layer compliance -- {len(llm_sample)} questions")
        results["baseline_dual_layer"] = run_prompt_ablation(
            "baseline_prompt", llm_sample, _SYSTEM_PROMPT_FULL
        )

        print(f"\n[A2-4] MSA-only responses -- {len(llm_sample)} questions")
        results["a2_4_msa_only"] = run_prompt_ablation(
            "a2_4", llm_sample, _SYSTEM_PROMPT_MSA
        )

    # -- Save BEFORE printing summary (crash-safe) ----------------------------
    results["summary"] = {
        "recall_baseline":  results["baseline_retrieval"]["recall_at_5"],
        "recall_a2_1":      results["a2_1_no_normalize"]["recall_at_5"],
        "recall_a2_2":      results["a2_2_dense_only"]["recall_at_5"],
        **({"bypass_baseline": results["baseline_guardrail"]["bypass_rate"],
            "bypass_a2_3":     results["a2_3_no_guardrail"]["bypass_rate"],
            "dual_baseline":   results["baseline_dual_layer"]["dual_layer_rate"],
            "dual_a2_4":       results["a2_4_msa_only"]["dual_layer_rate"],
           } if not args.skip_llm else {}),
    }

    out_path = Path(args.out)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Full results saved -> {out_path}")

    # -- Summary ---------------------------------------------------------------
    print("\n" + "=" * 60)
    print("ABLATION SUMMARY")
    print("=" * 60)

    b_r = results["baseline_retrieval"]["recall_at_5"]
    r1  = results["a2_1_no_normalize"]["recall_at_5"]
    r2  = results["a2_2_dense_only"]["recall_at_5"]
    print(f"\nRecall@5 (out of {len(all_questions)} questions):")
    print(f"  Baseline (hybrid + normalize) : {b_r:.2%}")
    print(f"  A2-1 No normalization         : {r1:.2%}  (delta {r1-b_r:+.2%})")
    print(f"  A2-2 Dense only               : {r2:.2%}  (delta {r2-b_r:+.2%})")

    if not args.skip_llm:
        b_g = results["baseline_guardrail"]["bypass_rate"]
        g3  = results["a2_3_no_guardrail"]["bypass_rate"]
        print(f"\nGuardrail bypass rate (out of {len(OFF_CURRICULUM_QUESTIONS)} off-curriculum Qs):")
        print(f"  Baseline (threshold=0.60)     : {b_g:.2%}")
        print(f"  A2-3 No guardrail (thresh=0)  : {g3:.2%}  (delta {g3-b_g:+.2%})")

        b_d = results["baseline_dual_layer"]["dual_layer_rate"]
        d4  = results["a2_4_msa_only"]["dual_layer_rate"]
        print(f"\nDual-layer compliance (out of {len(llm_sample)} questions):")
        print(f"  Baseline (full prompt)        : {b_d:.2%}")
        print(f"  A2-4 MSA-only                 : {d4:.2%}  (delta {d4-b_d:+.2%})")

    print("=" * 60)


if __name__ == "__main__":
    main()
