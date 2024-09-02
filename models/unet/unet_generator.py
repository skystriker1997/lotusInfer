import torch
import pnnx


#unet
model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet', in_channels=3, out_channels=1, init_features=32, pretrained=True)

x = torch.rand(1, 3, 256, 256)

opt_net = pnnx.export(model, "unet.pt", x)