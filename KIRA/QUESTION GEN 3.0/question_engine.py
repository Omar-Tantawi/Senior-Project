"""
Question generation engine — retrieval-grounded.

Pipeline per request:
  1. PLAN  — ask the local LLM for the real topics in this book (best-effort), and
             compute an even-coverage spread as a fallback. Assign each question slot
             a grounding passage from the SOURCE (a single book / uploaded doc only).
  2. GENERATE — for each slot, prompt the model with that exact passage (temp 0.4).
  3. VERIFY — score the question against its passage (evaluator). If it's weak or
             off-passage, regenerate once on a different passage / lower temperature
             and keep the best result.

Local model only (Ollama). Make sure it is running:
    ollama serve
    ollama create smartschool -f Modelfile     (or: ollama pull command-r7b-arabic)
"""

import json
import re
import os
import urllib.request
import urllib.error

from prompts   import build_prompt, build_outline_prompt
from evaluator import evaluate_question

OLLAMA_URL   = os.getenv("OLLAMA_URL",   "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "smartschool")
NUM_CTX      = int(os.getenv("OLLAMA_NUM_CTX", "4096"))

# ── Type normaliser ───────────────────────────────────────────────────────────

_TYPE_ALIASES: dict[str, str] = {
    "true_false": "true_false", "true-false": "true_false", "truefalse": "true_false",
    "true/false": "true_false", "tf": "true_false",
    "short_answer": "short_answer", "short-answer": "short_answer",
    "shortanswer": "short_answer", "short answer": "short_answer",
    "fill_blank": "fill_blank", "fill-blank": "fill_blank",
    "fill_in_the_blank": "fill_blank", "fill-in-the-blank": "fill_blank",
    "fill in the blank": "fill_blank", "fillblank": "fill_blank",
    "mcq": "mcq", "multiple_choice": "mcq", "multiple-choice": "mcq", "multiple choice": "mcq",
    "essay": "essay", "open_ended": "essay", "open-ended": "essay",
}


def _normalise_type(raw_type: str, fallback: str) -> str:
    return _TYPE_ALIASES.get(raw_type.strip(), fallback)


class QuestionEngine:
    def __init__(self):
        self._check_ollama()

    # ── public API ──────────────────────────────────────────────────────────────

    def generate(
        self,
        source,                          # CurriculumSource | MemSource (from retriever.py)
        num_questions:  int,
        question_types: list[str],
        bloom_levels:   list[str],
        difficulty:     str,
        language:       str,
        topic_hint:     str = "",
    ) -> list[dict]:
        if not source.indices:
            return []

        plan = self._plan(source, num_questions, question_types, bloom_levels,
                           topic_hint, language)

        questions = []
        for i, slot in enumerate(plan):
            print(f"  Q{i+1}/{len(plan)} [{slot['q_type']} | {slot['bloom']}]"
                  f" topic={slot['topic'][:40]!r} chunk#{slot['idx']}")
            q = self._generate_grounded(source, slot, difficulty, language, i + 1)
            if q:
                questions.append(q)
        return questions

    # ── planning ──────────────────────────────────────────────────────────────────

    def _plan(self, source, n, qtypes, blooms, topic_hint, language) -> list[dict]:
        topics: list[str] = []
        if topic_hint.strip():
            topics.append(topic_hint.strip())
        topics += self._outline_topics(source, n, language)

        coverage = source.even_coverage(n)          # query-free spread (fallback)
        used: set[int] = set()
        cov_ptr = 0
        plan = []

        for i in range(n):
            q_type = qtypes[i % len(qtypes)]
            bloom  = blooms[i % len(blooms)]
            topic_label = topics[i] if i < len(topics) else ""

            idx = None
            if topic_label:
                for cand in source.hybrid(topic_label, k=6):
                    if cand not in used:
                        idx = cand
                        break
            if idx is None:                          # even-coverage fallback
                while cov_ptr < len(coverage) and coverage[cov_ptr] in used:
                    cov_ptr += 1
                if cov_ptr < len(coverage):
                    idx = coverage[cov_ptr]; cov_ptr += 1
            if idx is None:                          # any unused chunk
                idx = next((c for c in source.indices if c not in used), None)
            if idx is None:                          # tiny doc: allow reuse
                idx = source.indices[i % len(source.indices)]

            used.add(idx)
            backups = [c for c in (source.hybrid(topic_label, k=10) if topic_label else [])
                       if c not in used][:2]
            if not backups:
                backups = [c for c in coverage if c not in used][:2]

            plan.append({"q_type": q_type, "bloom": bloom, "topic": topic_label,
                         "idx": idx, "backups": backups})
        return plan

    def _outline_topics(self, source, n, language) -> list[str]:
        samples = source.outline_samples(min(40, max(12, n * 3)))
        if not samples:
            return []
        system, user = build_outline_prompt(samples, n, language)
        raw = self._ollama_chat(system, user, temperature=0.3, num_predict=500)
        if not raw:
            return []
        return self._parse_topic_list(raw, n)

    @staticmethod
    def _parse_topic_list(raw: str, n: int) -> list[str]:
        raw = re.sub(r"```(?:json)?", "", raw).strip().strip("`").strip()
        topics = []
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        if match:
            try:
                arr = json.loads(match.group())
                topics = [str(t).strip() for t in arr if str(t).strip()]
            except json.JSONDecodeError:
                pass
        if not topics:                                # fallback: split bullet/numbered lines
            for line in raw.splitlines():
                line = re.sub(r'^[\s\-\*\d\.\)«"\'•]+', '', line).strip(' "\'،,')
                if len(line) > 2:
                    topics.append(line)
        # drop dupes, cap
        seen, out = set(), []
        for t in topics:
            key = t.lower()
            if key not in seen:
                seen.add(key); out.append(t)
        return out[:n]

    # ── grounded generation + verify/regenerate ────────────────────────────────

    def _generate_grounded(self, source, slot, difficulty, language, qnum) -> dict | None:
        attempts: list[tuple[int, dict, str]] = []

        ctx = source.context_for(slot["idx"])
        q = self._generate_one(ctx, slot, difficulty, language, qnum, temperature=0.4)
        if q:
            ev = evaluate_question(q, ctx, language)
            if self._good(ev):
                return self._finalise(q, ctx, slot, ev)
            attempts.append((ev["score"], q, ctx))

        # regenerate on backup passages
        for b in slot.get("backups", []):
            ctx2 = source.context_for(b)
            q2 = self._generate_one(ctx2, slot, difficulty, language, qnum, temperature=0.3)
            if q2:
                ev2 = evaluate_question(q2, ctx2, language)
                if self._good(ev2):
                    return self._finalise(q2, ctx2, slot, ev2)
                attempts.append((ev2["score"], q2, ctx2))

        # last resort: low-temp retry on the original passage
        q3 = self._generate_one(ctx, slot, difficulty, language, qnum, temperature=0.2)
        if q3:
            ev3 = evaluate_question(q3, ctx, language)
            attempts.append((ev3["score"], q3, ctx))

        if not attempts:
            return None
        score, bq, bctx = max(attempts, key=lambda a: a[0])
        return self._finalise(bq, bctx, slot, evaluate_question(bq, bctx, language))

    @staticmethod
    def _good(ev: dict) -> bool:
        return (
            ev["score"] >= 60
            and not ev["checks"]["structure"]["missing_fields"]
            and ev["checks"]["relevance"]["overlap"] >= 0.10
        )

    @staticmethod
    def _finalise(q: dict, ctx: str, slot: dict, ev: dict) -> dict:
        q["_context"] = ctx                 # main.py uses this for batch evaluation
        if slot.get("topic"):
            q.setdefault("topic", slot["topic"])
        return q

    def _generate_one(self, context, slot, difficulty, language, qnum, temperature) -> dict | None:
        system, user = build_prompt(
            context=context,
            question_type=slot["q_type"],
            bloom_level=slot["bloom"],
            difficulty=difficulty,
            language=language,
            question_num=qnum,
            topic=slot.get("topic", ""),
        )
        raw = self._ollama_chat(system, user, temperature=temperature, num_predict=800)
        if not raw:
            return None
        return self._parse_json(raw, slot["q_type"], slot["bloom"], difficulty)

    # ── Ollama transport ────────────────────────────────────────────────────────

    def _check_ollama(self):
        try:
            with urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=3) as resp:
                models = [m["name"] for m in json.loads(resp.read()).get("models", [])]
                base = OLLAMA_MODEL.split(":")[0]
                if any(base in m for m in models):
                    print(f"[QuestionEngine] Ollama ready — model: {OLLAMA_MODEL}")
                else:
                    print(f"[QuestionEngine] WARNING: model '{OLLAMA_MODEL}' not found. "
                          f"Run: ollama create {OLLAMA_MODEL} -f Modelfile")
        except urllib.error.URLError:
            print("[QuestionEngine] WARNING: Ollama is not running. Start it with: ollama serve")

    def _ollama_chat(self, system_prompt, user_prompt, temperature, num_predict) -> str | None:
        payload = json.dumps({
            "model": OLLAMA_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": num_predict,
                "num_ctx":     NUM_CTX,
            },
        }).encode("utf-8")
        req = urllib.request.Request(
            f"{OLLAMA_URL}/api/chat", data=payload,
            headers={"Content-Type": "application/json"}, method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                return json.loads(resp.read())["message"]["content"].strip()
        except urllib.error.URLError as e:
            print(f"[QuestionEngine] Ollama request failed: {e}")
            return None
        except (KeyError, json.JSONDecodeError) as e:
            print(f"[QuestionEngine] Bad response from Ollama: {e}")
            return None

    # ── JSON parsing for a single question ──────────────────────────────────────

    def _parse_json(self, raw, q_type, bloom_level, difficulty) -> dict | None:
        raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
        # strip stray LaTeX the model sometimes emits
        raw = re.sub(r"\\\(.*?\\\)",
                     lambda m: re.sub(r"\\text\{([^}]+)\}", r"\1", m.group())
                     .replace("\\(", "").replace("\\)", "").strip(), raw)
        raw = re.sub(r"\$[^$]+\$", "", raw)

        def _try(text):
            try:
                data = json.loads(text)
            except json.JSONDecodeError:
                return None
            data.setdefault("type", q_type)
            data.setdefault("bloom_level", bloom_level)
            data.setdefault("difficulty", difficulty)
            data["bloom_level"] = str(data["bloom_level"]).lower()
            data["difficulty"]  = str(data["difficulty"]).lower()
            data["type"] = _normalise_type(str(data["type"]).lower(), q_type)
            return data

        result = _try(raw)
        if result:
            return result
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            result = _try(match.group())
            if result:
                return result
        print(f"[QuestionEngine] Could not parse JSON:\n{raw[:300]}")
        return None
