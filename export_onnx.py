import torch
import torch.nn as nn
from torchvision import models

model = models.mobilenet_v2(weights=None)
num_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_features, 29)

model.load_state_dict(torch.load('mobilenet_v2_asl.pth', weights_only=True, map_location=torch.device('cpu')))

model.eval()

dummy_input = torch.randn(1, 3, 224, 224)

print("Exporting model to ONNX format...")
torch.onnx.export(
    model,
    dummy_input,
    "asl_classifier.onnx",
    export_params=True,
    opset_version=11,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    }
)

print("File 'asl_classifier.onnx' is ready.")