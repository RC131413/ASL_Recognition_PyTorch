import torch
import torchvision.models as models
import torch.nn as nn
import onnxruntime as ort
import numpy as np
import time

print("Loading models...")

model_pytorch = models.mobilenet_v2(weights=None)
model_pytorch.classifier[1] = nn.Linear(model_pytorch.classifier[1].in_features, 29)
model_pytorch.load_state_dict(torch.load('mobilenet_v2_asl.pth', weights_only=True, map_location='cpu'))
model_pytorch.eval()

ort_session = ort.InferenceSession("asl_classifier.onnx", providers=['CPUExecutionProvider'])

dummy_input_pytorch = torch.randn(1, 3, 224, 224)
dummy_input_onnx = dummy_input_pytorch.numpy()

with torch.no_grad():
    output_pytorch = model_pytorch(dummy_input_pytorch).numpy()

output_onnx = ort_session.run(None, {"input": dummy_input_onnx})[0]
diff = np.abs(output_pytorch - output_onnx).mean()
print(f"Difference: {diff:.8f}")

# WARMUP
print("Warming up CPUs...")
iterations = 100
warmup = 20

for _ in range(warmup):
    with torch.no_grad():
        _ = model_pytorch(dummy_input_pytorch)
    _ = ort_session.run(None, {"input": dummy_input_onnx})

print(f"Running speed test ({iterations} iterations)...")

# PyTorch Test
start_time = time.perf_counter()
for _ in range(iterations):
    with torch.no_grad():
        _ = model_pytorch(dummy_input_pytorch)
pytorch_time = (time.perf_counter() - start_time) / iterations

# ONNX Test
start_time = time.perf_counter()
for _ in range(iterations):
    _ = ort_session.run(None, {"input": dummy_input_onnx})
onnx_time = (time.perf_counter() - start_time) / iterations

print("\n--- RESULTS ---")
print(f"PyTorch time: {pytorch_time * 1000:.2f} ms")
print(f"ONNX time:    {onnx_time * 1000:.2f} ms")

if pytorch_time > onnx_time:
    speedup = pytorch_time / onnx_time
    print(f"ONNX is {speedup:.2f}x faster.")
else:
    speedup = onnx_time / pytorch_time
    print(f"PyTorch is {speedup:.2f}x faster.")