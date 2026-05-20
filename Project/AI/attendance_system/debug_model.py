"""Debug script to check ONNX model input/output shapes."""
import onnxruntime as ort
import numpy as np
import cv2

det_path = "models/det_10g.onnx"
rec_path = "models/w600k_r50.onnx"

print("=== Detection Model ===")
sess = ort.InferenceSession(det_path, providers=["CPUExecutionProvider"])
print(f"Inputs:")
for inp in sess.get_inputs():
    print(f"  {inp.name}: {inp.shape} ({inp.type})")
print(f"Outputs ({len(sess.get_outputs())}):")
for i, out in enumerate(sess.get_outputs()):
    print(f"  [{i}] {out.name}: {out.shape} ({out.type})")

# Run with dummy input to see actual shapes
dummy = np.zeros((1, 3, 640, 640), dtype=np.float32)
outputs = sess.run(None, {sess.get_inputs()[0].name: dummy})
print(f"\nActual output shapes:")
for i, o in enumerate(outputs):
    print(f"  [{i}] {o.shape}")

print(f"\n=== Recognition Model ===")
sess2 = ort.InferenceSession(rec_path, providers=["CPUExecutionProvider"])
print(f"Inputs:")
for inp in sess2.get_inputs():
    print(f"  {inp.name}: {inp.shape} ({inp.type})")
print(f"Outputs:")
for out in sess2.get_outputs():
    print(f"  {out.name}: {out.shape} ({out.type})")
