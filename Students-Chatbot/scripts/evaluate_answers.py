"""
LLM-as-judge evaluation script.

For each question in eval_questions.json:
  1. Runs the full chatbot pipeline (retrieve → LLM) to get the system answer.
  2. Asks a judge LLM to score the answer against the expected answer (0–3).
  3. Saves per-question results and a per-subject / overall summary.

Score rubric (0–3):
  0 — Wrong or completely irrelevant
  1 — Partially correct (key idea present but incomplete or contains errors)
  2 — Mostly correct (main facts right, minor gaps acceptable)
  3 — Fully correct (matches or is equivalent to the expected answer)

Scores are also reported as a 0–1 mean and as "accuracy" (% of questions ≥ 2).

Usage:
    python scripts/evaluate_answers.py
    python scripts/evaluate_answers.py --judge-model llama3.1:8b
    python scripts/evaluate_answers.py --sample 30 --seed 42
    python scripts/evaluate_answers.py --eval eval_questions.json --out eval_results.json

Resume: re-running skips questions that already have a score in the output file.

Notes:
  - Use a model DIFFERENT from the chatbot model as the judge for unbiased scoring.
    Recommended: a larger instruction-tuned model available via Ollama
    (e.g. llama3.1:8b, aya:8b, mistral:7b).  The chatbot model is used as
    fallback if no --judge-model is supplied, but results will be less reliable.
  - Only one process may access the Qdrant storage at a time.
    Shut down the FastAPI server before running this script.
"""
import sys
import json
import re
import argparse
import random
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

import ollama
from tqdm import tqdm

from chatbot.config import LLM_MODEL
from chatbot.retrieval.retriever import retrieve
from chatbot.llm.client import chat, _NO_CONTEXT_REPLY

# ── Constants ──────────────────────────────────────────────────────────────────

_JUDGE_PROMPT = """\
أنت محكّم خبير في تقييم إجابات الأنظمة التعليمية.

السؤال:
{question}

الإجابة المرجعية (الصحيحة):
{expected}

إجابة النظام:
{actual}

قيّم إجابة النظام على المقياس التالي:
0 = خاطئة تماماً أو غير ذات صلة بالسؤال
1 = صحيحة جزئياً (الفكرة الأساسية موجودة لكن الإجابة ناقصة أو تحتوي على أخطاء)
2 = صحيحة في معظمها (المعلومات الجوهرية صحيحة مع اختلافات بسيطة مقبولة)
3 = صحيحة تماماً (تطابق الإجابة المرجعية أو تكافئها في المعنى)

اكتب سطراً واحداً فقط بهذا الشكل:
الدرجة: [0 أو 1 أو 2 أو 3]
"""

_SCORE_RE = re.compile(r"الدرجة\s*:\s*([0-3])")


# ── Core functions ─────────────────────────────────────────────────────────────

def get_chatbot_answer(question: str, grade: str | None = None) -> str:
    """Run the full retrieval + generation pipeline for one question."""
    chunks = retrieve(question, grade=grade)
    return chat(question, chunks)


def judge_answer(
    question: str,
    expected: str,
    actual: str,
    judge_model: str,
) -> int | None:
    """
    Ask the judge LLM to score `actual` vs `expected`.
    Returns an integer 0–3, or None if the judge response cannot be parsed.
    """
    prompt = _JUDGE_PROMPT.format(
        question=question,
        expected=expected,
        actual=actual,
    )
    try:
        resp = ollama.chat(
            model=judge_model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.0},
        )
        content = resp["message"]["content"].strip()
        match = _SCORE_RE.search(content)
        if match:
            return int(match.group(1))
        # fallback: look for a bare digit
        digits = re.findall(r"[0-3]", content)
        if digits:
            return int(digits[0])
        return None
    except Exception as e:
        print(f"  [judge warn] {e}")
        return None


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval",        default="eval_questions.json")
    parser.add_argument("--out",         default="eval_results.json")
    parser.add_argument("--judge-model", default=LLM_MODEL,
                        help="Ollama model used as judge (default: chatbot model). "
                             "Use a different/stronger model for unbiased evaluation.")
    parser.add_argument("--sample",      type=int, default=None,
                        help="Evaluate a random sample instead of all 150 questions.")
    parser.add_argument("--seed",        type=int, default=42)
    args = parser.parse_args()

    if args.judge_model == LLM_MODEL:
        print(
            f"[warn] Judge model is the same as the chatbot model ({LLM_MODEL}).\n"
            "       Results will be less reliable. Consider --judge-model <stronger-model>.\n"
        )

    random.seed(args.seed)

    with open(args.eval, encoding="utf-8") as f:
        all_questions: list[dict] = json.load(f)

    if args.sample:
        all_questions = random.sample(all_questions, min(args.sample, len(all_questions)))
        print(f"Evaluating a random sample of {len(all_questions)} questions.\n")
    else:
        print(f"Evaluating all {len(all_questions)} questions.\n")

    # Resume: load existing results and skip already-scored questions.
    out_path = Path(args.out)
    if out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            saved = json.load(f)
        results: list[dict] = saved.get("results", [])
        scored_ids = {r["id"] for r in results}
        print(f"Resuming — {len(scored_ids)} questions already scored.\n")
    else:
        results = []
        scored_ids: set[int] = set()

    # ── Evaluation loop ────────────────────────────────────────────────────────
    for q in tqdm(all_questions, desc="Evaluating"):
        if q["id"] in scored_ids:
            continue

        chatbot_answer = get_chatbot_answer(q["question"], grade=q.get("grade"))

        # If the guardrail blocked the question, score 0 without calling judge.
        if _NO_CONTEXT_REPLY in chatbot_answer:
            score = 0
            judge_reasoning = "guardrail_blocked"
        else:
            score = judge_answer(
                question=q["question"],
                expected=q["expected_answer"],
                actual=chatbot_answer,
                judge_model=args.judge_model,
            )
            judge_reasoning = "parsed" if score is not None else "parse_failed"
            if score is None:
                score = 0  # treat unparseable judge responses as 0

        results.append({
            "id":              q["id"],
            "subject":         q["subject"],
            "grade":           q.get("grade"),
            "question":        q["question"],
            "expected_answer": q["expected_answer"],
            "chatbot_answer":  chatbot_answer,
            "score":           score,          # 0–3
            "judge_note":      judge_reasoning,
        })

        # Save after every question (crash-safe).
        _save(out_path, results, args)

    # ── Summary ────────────────────────────────────────────────────────────────
    _save(out_path, results, args)
    _print_summary(results)


def _save(out_path: Path, results: list[dict], args: argparse.Namespace) -> None:
    by_subject: dict[str, list[int]] = defaultdict(list)
    for r in results:
        by_subject[r["subject"]].append(r["score"])

    subject_summary = {}
    for subj, scores in by_subject.items():
        subject_summary[subj] = {
            "n":        len(scores),
            "mean":     round(sum(scores) / len(scores) / 3, 4),   # 0–1
            "accuracy": round(sum(s >= 2 for s in scores) / len(scores), 4),  # % ≥ 2
            "score_dist": {
                "0": scores.count(0),
                "1": scores.count(1),
                "2": scores.count(2),
                "3": scores.count(3),
            },
        }

    all_scores = [r["score"] for r in results]
    overall = {
        "n":          len(all_scores),
        "mean":       round(sum(all_scores) / len(all_scores) / 3, 4) if all_scores else 0,
        "accuracy":   round(sum(s >= 2 for s in all_scores) / len(all_scores), 4) if all_scores else 0,
        "judge_model": args.judge_model,
        "chatbot_model": LLM_MODEL,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"overall": overall, "by_subject": subject_summary, "results": results},
                  f, ensure_ascii=False, indent=2)


def _print_summary(results: list[dict]) -> None:
    if not results:
        print("No results to summarise.")
        return

    by_subject: dict[str, list[int]] = defaultdict(list)
    for r in results:
        by_subject[r["subject"]].append(r["score"])

    print("\n" + "=" * 62)
    print("EVALUATION SUMMARY  (score 0-3 -> normalised to 0-1)")
    print("=" * 62)
    print(f"  {'Subject':<15} {'N':>5}  {'Mean (0-1)':>10}  {'Accuracy (>=2)':>14}")
    print(f"  {'-'*15} {'-'*5}  {'-'*10}  {'-'*14}")

    all_scores: list[int] = []
    for subj, scores in sorted(by_subject.items()):
        mean     = sum(scores) / len(scores) / 3
        accuracy = sum(s >= 2 for s in scores) / len(scores)
        print(f"  {subj:<15} {len(scores):>5}  {mean:>10.2%}  {accuracy:>14.2%}")
        all_scores.extend(scores)

    print(f"  {'='*15} {'='*5}  {'='*10}  {'='*14}")
    overall_mean = sum(all_scores) / len(all_scores) / 3
    overall_acc  = sum(s >= 2 for s in all_scores) / len(all_scores)
    print(f"  {'OVERALL':<15} {len(all_scores):>5}  {overall_mean:>10.2%}  {overall_acc:>14.2%}")
    print("=" * 62)

    blocked = sum(1 for r in results if r["judge_note"] == "guardrail_blocked")
    failed  = sum(1 for r in results if r["judge_note"] == "parse_failed")
    if blocked:
        print(f"\n  Guardrail blocked (scored 0 automatically): {blocked}")
    if failed:
        print(f"  Judge parse failures (scored 0):             {failed}")


if __name__ == "__main__":
    main()
