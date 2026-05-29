"""
CLI chat for quick local testing without starting the API server.

Usage:
    python scripts/chat.py
    python scripts/chat.py --grade 11 --subject الفيزياء
"""
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from chatbot.retrieval.retriever import retrieve
from chatbot.llm.client import chat


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--grade", default=None, help="Filter by grade: 10, 11, 12")
    parser.add_argument("--subject", default=None, help="Filter by subject name (Arabic)")
    args = parser.parse_args()

    print("محادثة مع المساعد التعليمي السوري — اكتب 'خروج' للإنهاء\n")

    while True:
        try:
            question = input("أنت: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nوداعاً!")
            break

        if question in ("خروج", "exit", "quit"):
            print("وداعاً!")
            break

        if not question:
            continue

        chunks = retrieve(question)
        if args.grade:
            chunks = [c for c in chunks if c.get("grade") == args.grade]
        if args.subject:
            chunks = [c for c in chunks if args.subject in c.get("subject", "")]

        if not chunks:
            print("المساعد: هذا السؤال خارج المنهج الذي لديّ.\n")
            continue

        print(f"\n[مصادر: {', '.join(set(c['subject'] + ' ص' + str(c['page_num']) for c in chunks))}]")
        print("\nالمساعد:")
        answer = chat(question, chunks)
        print(answer)
        print()


if __name__ == "__main__":
    main()
