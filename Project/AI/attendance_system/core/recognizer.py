import os
import numpy as np
from pathlib import Path


class FaceRecognizer:
    """Matches face embeddings against enrolled student database."""

    def __init__(self, embeddings_dir: str, similarity_threshold: float = 0.4,
                 unknown_threshold: float = 0.3):
        self.embeddings_dir = Path(embeddings_dir)
        self.similarity_threshold = similarity_threshold
        self.unknown_threshold = unknown_threshold

        # student_id -> {"name": str, "embeddings": np.ndarray shape (N, 512)}
        self.database: dict = {}
        self.load_database()

    def load_database(self):
        """Load all enrolled student embeddings from disk."""
        self.database = {}
        if not self.embeddings_dir.exists():
            self.embeddings_dir.mkdir(parents=True, exist_ok=True)
            return

        for student_dir in self.embeddings_dir.iterdir():
            if not student_dir.is_dir():
                continue

            emb_file = student_dir / "embeddings.npy"
            name_file = student_dir / "name.txt"

            if emb_file.exists() and name_file.exists():
                student_id = student_dir.name
                embeddings = np.load(str(emb_file))
                name = name_file.read_text().strip()
                self.database[student_id] = {
                    "name": name,
                    "embeddings": embeddings,
                }

        print(f"[Recognizer] Loaded {len(self.database)} enrolled students")

    def enroll_student(self, student_id: str, student_name: str, embeddings: list[np.ndarray]):
        """Save a student's face embeddings to disk."""
        student_dir = self.embeddings_dir / student_id
        student_dir.mkdir(parents=True, exist_ok=True)

        emb_array = np.stack(embeddings)  # (N, 512)
        np.save(str(student_dir / "embeddings.npy"), emb_array)
        (student_dir / "name.txt").write_text(student_name)

        self.database[student_id] = {
            "name": student_name,
            "embeddings": emb_array,
        }
        print(f"[Recognizer] Enrolled {student_name} (ID: {student_id}) with {len(embeddings)} photos")

    def recognize(self, embedding: np.ndarray) -> tuple[str | None, str | None, float]:
        """Match a face embedding against the database.

        Returns:
            (student_id, student_name, confidence) or (None, None, confidence)
        """
        if not self.database:
            return None, None, 0.0

        best_id = None
        best_name = None
        best_score = -1.0

        for student_id, data in self.database.items():
            # Cosine similarity between query and each enrolled embedding
            similarities = np.dot(data["embeddings"], embedding) / (
                np.linalg.norm(data["embeddings"], axis=1) * np.linalg.norm(embedding)
            )
            max_sim = float(np.max(similarities))

            if max_sim > best_score:
                best_score = max_sim
                best_id = student_id
                best_name = data["name"]

        if best_score >= self.similarity_threshold:
            return best_id, best_name, best_score
        elif best_score >= self.unknown_threshold:
            return None, None, best_score  # Low confidence, mark unknown
        else:
            return None, None, best_score

    def get_enrolled_count(self) -> int:
        return len(self.database)

    def is_enrolled(self, student_id: str) -> bool:
        return student_id in self.database

    def delete_student(self, student_id: str):
        """Remove a student from the database."""
        student_dir = self.embeddings_dir / student_id
        if student_dir.exists():
            import shutil
            shutil.rmtree(student_dir)
        self.database.pop(student_id, None)
        print(f"[Recognizer] Deleted student {student_id}")
