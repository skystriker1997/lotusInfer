import torch
import pnnx


#resnet
model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
x = torch.rand(1, 3, 224, 224)

opt_net = pnnx.export(model, "resnet18.pt", x)