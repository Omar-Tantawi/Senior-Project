"""
Generates 50 evaluation questions per subject (Physics, Math, Chemistry)
from the indexed curriculum chunks using the local LLM.

Usage:
    python scripts/generate_eval_questions.py
    python scripts/generate_eval_questions.py --per-subject 50 --out eval_questions.json

Resume: re-running automatically skips subjects that already have enough questions.
"""
import sys
import json
import random
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import ollama
from qdrant_client import QdrantClient
from tqdm import tqdm

from chatbot.config import QDRANT_DIR, COLLECTION_NAME, LLM_MODEL

SUBJECT_GROUPS = {
    "physics":   ["sci-physics", "physics-Sci", "physics-sci"],
    "math":      ["sci-math-1", "sci-math-2", "math1", "math2"],
    "chemistry": ["sci-chemistry", "chemistry-Sci", "chemistry-sci"],
}

_GEN_PROMPT = """\
أنت مساعد تعليمي. سيُعطى لك مقطع من كتاب مدرسي سوري.
مهمتك: اكتب سؤالاً واحداً واضحاً باللغة العربية يمكن الإجابة عليه مباشرةً من المقطع،
ثم اكتب الإجابة الصحيحة الكاملة من المقطع نفسه.

القواعد:
- السؤال يجب أن يكون محدداً وقابلاً للقياس (لا أسئلة رأي أو عامة).
- الإجابة يجب أن تكون مستخرجة حرفياً أو ملخصة من المقطع فقط.
- اكتب الإجابة بشكل كامل وواضح.
- لا تكتب أي شيء آخر خارج التنسيق التالي:

السؤال: [السؤال هنا]
الجواب: [الجواب هنا]

المقطع:
{chunk}
"""


def _fetch_chunks(client: QdrantClient, subjects: list[str]) -> list[dict]:
    chunks = []
    offset = None
    while True:
        results, offset = client.scroll(
            COLLECTION_NAME,
            limit=256,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for r in results:
            if r.payload.get("subject") in subjects:
                text = r.payload.get("text", "")
                if len(text.split()) >= 60:   # skip very short chunks
                    chunks.append(r.payload)
        if offset is None:
            break
    return chunks


def _generate_qa(chunk_text: str) -> tuple[str, str] | None:
    prompt = _GEN_PROMPT.format(chunk=chunk_text[:1200])
    try:
        resp = ollama.chat(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.3},
        )
        content = resp["message"]["content"].strip()
        q_line = next((l for l in content.splitlines() if l.startswith("السؤال:")), None)
        a_line = next((l for l in content.splitlines() if l.startswith("الجواب:")), None)
        if not q_line or not a_line:
            return None
        question = q_line.replace("السؤال:", "").strip()
        answer   = a_line.replace("الجواب:", "").strip()
        if len(question) < 10 or len(answer) < 10:
            return None
        return question, answer
    except Exception as e:
        print(f"  [warn] LLM error: {e}")
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-subject", type=int, default=50)
    parser.add_argument("--out", default="eval_questions.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    client = QdrantClient(path=str(QDRANT_DIR))
    out_path = Path(args.out)

    # Resume: load existing questions and count per subject
    if out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            all_questions = json.load(f)
        print(f"Resuming — loaded {len(all_questions)} existing questions")
    else:
        all_questions = []

    existing_counts = {}
    for q in all_questions:
        existing_counts[q["subject"]] = existing_counts.get(q["subject"], 0) + 1

    q_id = (max(q["id"] for q in all_questions) + 1) if all_questions else 1

    for subject_key, subject_tags in SUBJECT_GROUPS.items():
        print(f"\n=== {subject_key.upper()} ===")
        already = existing_counts.get(subject_key, 0)
        if already >= args.per_subject:
            print(f"  Skipping — already have {already} questions")
            continue

        chunks = _fetch_chunks(client, subject_tags)
        print(f"  Found {len(chunks)} eligible chunks")

        if len(chunks) < args.per_subject:
            print(f"  [warn] only {len(chunks)} chunks — will generate fewer questions")

        random.shuffle(chunks)
        generated = already
        still_needed = args.per_subject - already

        with tqdm(total=args.per_subject, initial=already, desc=f"  Generating") as pbar:
            for chunk in chunks:
                if generated - already >= still_needed:
                    break
                result = _generate_qa(chunk.get("text", ""))
                if result is None:
                    continue
                question, answer = result
                all_questions.append({
                    "id": q_id,
                    "subject": subject_key,
                    "grade": chunk.get("grade"),
                    "book": chunk.get("book") or chunk.get("source"),
                    "page_num": chunk.get("page_num"),
                    "question": question,
                    "expected_answer": answer,
                })
                q_id += 1
                generated += 1
                pbar.update(1)

                # save after every question so progress survives crashes
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(all_questions, f, ensure_ascii=False, indent=2)

        print(f"  Done: {generated} questions generated")

    print(f"\nTotal: {len(all_questions)} questions saved to {out_path.resolve()}")


if __name__ == "__main__":
    main()
