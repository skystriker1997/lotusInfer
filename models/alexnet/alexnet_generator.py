import torch
import pnnx

# alexnet
alexnet = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained=True)
alexnet_input = torch.rand(1, 3, 224, 224)

opt_net = pnnx.export(alexnet, "AlexNet.pt", alexnet_input)
