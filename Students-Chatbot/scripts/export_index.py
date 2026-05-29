"""
Run this on the other device to export the Grade 10+11 index.
Produces: grade_10_11_export.pkl

Usage:
    python scripts/export_index.py
    python scripts/export_index.py --qdrant-path path/to/qdrant_storage
"""
import argparse
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qdrant-path", default="qdrant_storage")
    parser.add_argument("--collection", default="curriculum")
    parser.add_argument("--out", default="grade_10_11_export.pkl")
    args = parser.parse_args()

    from qdrant_client import QdrantClient

    client = QdrantClient(path=args.qdrant_path)
    total = client.count(args.collection).count
    print(f"Exporting {total} points from '{args.collection}'…")

    points = []
    offset = None
    while True:
        batch, offset = client.scroll(
            args.collection,
            limit=256,
            offset=offset,
            with_payload=True,
            with_vectors=True,
        )
        for p in batch:
            points.append({"vector": p.vector, "payload": p.payload})
        print(f"  fetched {len(points)}/{total}", end="\r")
        if offset is None:
            break

    out_path = Path(args.out)
    with open(out_path, "wb") as f:
        pickle.dump(points, f)

    print(f"\nExported {len(points)} points → {out_path.resolve()}")


if __name__ == "__main__":
    main()
