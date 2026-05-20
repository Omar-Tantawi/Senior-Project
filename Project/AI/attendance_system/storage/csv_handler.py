import csv
import os
from datetime import date
from pathlib import Path


class CSVHandler:
    """Saves attendance reports to CSV files."""

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save_report(self, report: dict) -> str:
        """Save an attendance report to a CSV file.

        Args:
            report: dict with keys "date", "present", "absent"

        Returns:
            Path to the saved CSV file.
        """
        report_date = report.get("date", date.today().isoformat())
        filename = f"attendance_{report_date}.csv"
        filepath = self.output_dir / filename

        all_records = report.get("present", []) + report.get("absent", [])
        all_records.sort(key=lambda r: r["student_id"])

        fieldnames = [
            "student_id", "student_name", "status",
            "first_seen", "last_seen", "best_confidence", "detection_count",
        ]

        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for record in all_records:
                row = {field: record.get(field, "") for field in fieldnames}
                writer.writerow(row)

        print(f"[CSV] Saved attendance report to {filepath}")
        return str(filepath)
