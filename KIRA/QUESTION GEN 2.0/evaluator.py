"""
Question Quality Evaluator.

Automatically scores generated questions across four dimensions:
  1. Structure   (30 pts) — all required fields present and non-empty
  2. Answer      (25 pts) — answer is valid for the question type
  3. Content     (25 pts) — question length and readability
  4. Relevance   (20 pts) — question relates to the source text

Total: 100 pts
  90-100 → Excellent
  75-89  → Good
  60-74  → Acceptable
  <60    → Poor (consider regenerating)
"""

import re
from typing import Optional

# ── Required fields per question type ────────────────────────────────────────

_REQUIRED_FIELDS: dict[str, list[str]] = {
    "mcq":          ["question", "options", "correct_answer", "explanation"],
    "true_false":   ["question", "correct_answer", "explanation"],
    "short_answer": ["question", "model_answer"],
    "fill_blank":   ["question", "correct_answer"],
    "essay":        ["question", "key_points"],
}

# Arabic stop words to exclude from relevance calculation
_ARABIC_STOPWORDS = {
    "من", "في", "على", "إلى", "و", "أو", "هو", "هي", "هم", "هن",
    "أن", "إن", "كان", "كانت", "يكون", "تكون", "ما", "لا", "عن",
    "مع", "بين", "قد", "كل", "هذا", "هذه", "ذلك", "تلك", "التي",
    "الذي", "الذين", "اللواتي", "حيث", "كما", "إذا", "لكن", "بل",
}

_ENGLISH_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "in", "of", "and",
    "or", "to", "that", "this", "it", "be", "has", "have", "had",
    "for", "on", "at", "by", "with", "as", "its", "their", "they",
    "he", "she", "we", "you", "i", "not", "but", "so", "if", "do",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _is_arabic_text(text: str) -> bool:
    arabic_chars = len(re.findall(r'[؀-ۿ]', text))
    return arabic_chars / max(len(text), 1) > 0.2


def _normalize_options(options) -> dict:
    """
    The model sometimes returns options as a list instead of a dict.
    Normalize both formats to {"A": "text", "B": "text", ...}
    """
    if isinstance(options, dict):
        return options
    if isinstance(options, list):
        result = {}
        letters = ["A", "B", "C", "D"]
        for i, opt in enumerate(options):
            key = letters[i] if i < len(letters) else str(i)
            if isinstance(opt, dict):
                # e.g. {"option": "text", "correct": true} or {"text": "...", "label": "A"}
                text = (opt.get("option") or opt.get("text") or
                        opt.get("value") or opt.get("content") or str(opt))
            else:
                text = str(opt)
            result[key] = text
        return result
    return {}


def _word_set(text: str) -> set[str]:
    """Lowercase word set with stop words removed."""
    words  = set(re.findall(r'\w+', text.lower()))
    words -= _ARABIC_STOPWORDS
    words -= _ENGLISH_STOPWORDS
    return words


def _overlap_score(text_a: str, text_b: str) -> float:
    """Jaccard-like overlap between meaningful words in two texts. Returns 0.0–1.0."""
    set_a = _word_set(text_a)
    set_b = _word_set(text_b)
    if not set_a or not set_b:
        return 0.0
    intersection = set_a & set_b
    # Use size of query (question) as denominator so short questions aren't penalised
    return len(intersection) / len(set_a)


# ── Single-question evaluator ─────────────────────────────────────────────────

def evaluate_question(
    question: dict,
    context:  str  = "",
    language: str  = "en",
) -> dict:
    """
    Score a single question dict.

    Returns:
        {
          "score":   int (0-100),
          "grade":   str ("Excellent" | "Good" | "Acceptable" | "Poor"),
          "passed":  bool,
          "checks": {
            "structure":   {"score": int, "max": 30, "missing_fields": list},
            "answer":      {"score": int, "max": 25, "detail": str},
            "content":     {"score": int, "max": 25, "word_count": int},
            "relevance":   {"score": int, "max": 20, "overlap": float},
            "language_ok": bool,
          }
        }
    """
    q_type = question.get("type", "unknown")
    checks = {}
    total  = 0

    # ── 1. Structure check (30 pts) ───────────────────────────────────────────
    required    = _REQUIRED_FIELDS.get(q_type, ["question"])
    missing     = [f for f in required if not question.get(f)]
    struct_pts  = round(30 * (1 - len(missing) / max(len(required), 1)))
    checks["structure"] = {
        "score":          struct_pts,
        "max":            30,
        "missing_fields": missing,
    }
    total += struct_pts

    # ── 2. Answer validity (25 pts) ───────────────────────────────────────────
    ans_pts = 0
    detail  = ""

    if q_type == "mcq":
        options = _normalize_options(question.get("options", {}))
        correct = str(question.get("correct_answer", "")).strip().upper()
        all_opts_present = all(k in options and options[k] for k in ["A", "B", "C", "D"])
        correct_valid    = correct in {"A", "B", "C", "D"}
        if all_opts_present:
            ans_pts += 15
        if correct_valid:
            ans_pts += 10
        detail = f"options={'OK' if all_opts_present else 'FAIL'}, correct_answer={'OK' if correct_valid else 'FAIL'}"

    elif q_type == "true_false":
        correct = str(question.get("correct_answer", "")).strip()
        valid   = correct in {"True", "False", "صح", "خطأ", "true", "false"}
        ans_pts = 25 if valid else 0
        detail  = f"correct_answer={'OK' if valid else f'FAIL (got: {correct!r})'}"

    elif q_type == "short_answer":
        model       = question.get("model_answer", "")
        word_count  = len(model.split())
        if word_count >= 10:
            ans_pts = 25
        elif word_count >= 5:
            ans_pts = 15
        elif word_count > 0:
            ans_pts = 5
        detail = f"model_answer word count: {word_count}"

    elif q_type == "fill_blank":
        correct  = question.get("correct_answer", "")
        q_text   = question.get("question", "")
        has_blank = "_" in q_text
        has_ans   = bool(correct and correct.strip())
        ans_pts   = (15 if has_blank else 0) + (10 if has_ans else 0)
        detail    = f"blank_in_question={'OK' if has_blank else 'FAIL'}, answer_provided={'OK' if has_ans else 'FAIL'}"

    elif q_type == "essay":
        key_points = question.get("key_points", [])
        count      = len([p for p in key_points if p and str(p).strip()])
        ans_pts    = min(25, count * 6)   # 6 pts per point, max 25
        detail     = f"key_points count: {count}"

    checks["answer"] = {"score": ans_pts, "max": 25, "detail": detail}
    total += ans_pts

    # ── 3. Content quality (25 pts) ───────────────────────────────────────────
    q_text    = question.get("question", "")
    word_count = len(q_text.split())

    if word_count >= 8 and word_count <= 80:
        content_pts = 25
    elif word_count >= 4:
        content_pts = 15
    elif word_count > 0:
        content_pts = 5
    else:
        content_pts = 0

    checks["content"] = {
        "score":      content_pts,
        "max":        25,
        "word_count": word_count,
    }
    total += content_pts

    # ── 4. Context relevance (20 pts) ─────────────────────────────────────────
    if context and context.strip():
        # Build search text: question + options (for MCQ)
        search_text = q_text
        if q_type == "mcq":
            options = _normalize_options(question.get("options", {}))
            search_text += " " + " ".join(options.values())

        overlap      = _overlap_score(search_text, context)
        rel_pts      = min(20, round(overlap * 50))   # 50% overlap → full 20 pts
    else:
        overlap = 0.0
        rel_pts = 15   # no context provided — partial credit (can't verify)

    checks["relevance"] = {
        "score":   rel_pts,
        "max":     20,
        "overlap": round(overlap, 3),
    }
    total += rel_pts

    # ── 5. Language sanity check (no pts, just a warning flag) ────────────────
    if language == "ar":
        arabic_ok = _is_arabic_text(q_text)
    else:
        # For English, just make sure it's not empty
        arabic_ok = True   # field named language_ok below

    checks["language_ok"] = (arabic_ok if language == "ar" else bool(q_text.strip()))

    # ── Final score ───────────────────────────────────────────────────────────
    score = min(100, total)
    # Penalise if language doesn't match (Arabic mode but no Arabic text)
    if language == "ar" and not checks["language_ok"]:
        score = max(0, score - 15)

    return {
        "score":  score,
        "grade":  _grade_label(score),
        "passed": score >= 60,
        "checks": checks,
    }


# ── Batch evaluator ───────────────────────────────────────────────────────────

def evaluate_batch(
    questions: list[dict],
    contexts:  Optional[list[str]] = None,
    language:  str = "en",
) -> dict:
    """
    Evaluate a list of questions.

    Args:
        questions: List of question dicts.
        contexts:  Parallel list of source context strings used to generate each question.
                   Pass None if contexts are not available.
        language:  "ar" or "en".

    Returns:
        {
          "questions":         list of question dicts with "quality" field added,
          "summary": {
            "average_score":   float,
            "min_score":       int,
            "max_score":       int,
            "questions_passed":int,
            "total":           int,
            "pass_rate":       float  (0-100),
            "overall_grade":   str,
          }
        }
    """
    if contexts is None:
        contexts = [""] * len(questions)

    evaluated = []
    for q, ctx in zip(questions, contexts):
        result = evaluate_question(q, ctx, language)
        q_copy = dict(q)
        q_copy["quality"] = result
        evaluated.append(q_copy)

    scores  = [q["quality"]["score"] for q in evaluated]
    passed  = [q for q in evaluated if q["quality"]["passed"]]
    avg     = round(sum(scores) / len(scores), 1) if scores else 0.0

    return {
        "questions": evaluated,
        "summary": {
            "average_score":    avg,
            "min_score":        min(scores) if scores else 0,
            "max_score":        max(scores) if scores else 0,
            "questions_passed": len(passed),
            "total":            len(questions),
            "pass_rate":        round(len(passed) / len(questions) * 100, 1) if questions else 0.0,
            "overall_grade":    _grade_label(avg),
        },
    }


def _grade_label(score: float) -> str:
    if score >= 90:
        return "Excellent"
    if score >= 75:
        return "Good"
    if score >= 60:
        return "Acceptable"
    return "Poor"
