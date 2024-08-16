import torch
import pnnx


model = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained=True)
x = torch.rand(1, 3, 224, 224)


opt_net = pnnx.export(model, "AlexNet.pt", x)

