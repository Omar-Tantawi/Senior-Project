from ultralytics import YOLO
from pathlib import Path

if __name__ == "__main__":
    BASE = Path(__file__).parent
    model = YOLO(str(BASE / "weights/sms_v1/weights/best.pt"))
    results = model.val(data=str(BASE / "sms/data.yaml"), split="test", verbose=True)
    r = results.results_dict

    print("--- TEST SET RESULTS ---")
    print(f"mAP@50     : {r['metrics/mAP50(B)']:.4f}")
    print(f"mAP@50-95  : {r['metrics/mAP50-95(B)']:.4f}")
    print(f"Precision  : {r['metrics/precision(B)']:.4f}")
    print(f"Recall     : {r['metrics/recall(B)']:.4f}")
    print("\nPer-class mAP@50:")
    for i, name in enumerate(["High Attention", "Low Attention"]):
        print(f"  {name}: {results.box.ap50[i]:.4f}")
