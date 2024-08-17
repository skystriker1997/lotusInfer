import torch
import torchvision.models as models
import pnnx


#resnet
net = models.resnet18(pretrained=True)
x = torch.rand(1, 3, 224, 224)

opt_net = pnnx.export(net, "resnet18.pt", x)