![logo](logo/logo.png)

## Overview

lotusInfer is a lightweight deep learning inference framework with CUDA acceleration. Built from the ground up, it offers high-performance CUDA operators without relying on cuBLAS or cuDNN. The long-term vision is to create a powerful framework capable of supporting cutting-edge Large Language Model (LLM) inference.


## Key Features
- **Independent Implementation**: All operators are developed from scratch, ensuring full control and customization
- **High Performance**: Fine-grained operators utilize both CUDA cores and Tensor Cores to maximize NVIDIA GPU architecture potential
- **Operator Fusion**: Activation and addition functions are embedded into preceding operators, optimizing the computation/memory access ratio
- **Clear Architecture**: Straightforward and transparent graph construction for easy understanding and modification


## Implemented Operators
- 2D Convolution
- GEMM
- Pooling
- More operators (including self-attention & multi-head attention) in development



## Development Environment and Dependencies
- **Operating System**: WSL Ubuntu 22.04.4 
- **C++ Standard**: C++20
- **CUDA Compute Capability**: 8.6 or above
- **Neural Network Model Format**: [PNNX](https://github.com/Tencent/ncnn/tree/master/tools/pnnx)
- **Multi-dimensional Array Library**: [xtensor3](https://xtensor.readthedocs.io/en/latest/)
- **Logging library**: [spdlog](https://github.com/gabime/spdlog)
- **Image Processing Library**: OpenCV


## Install
```bash
$ git clone https://github.com/skystriker1997/lotusInfer.git
$ cd lotusInfer && mkdir build && cd build
$ cmake .. && make -j
```

## U-Net Demo
![CT scan](models/unet/TCGA_CS_4944.png)  
source of image: [mateuszbuda/brain-segmentation-pytorch](https://github.com/mateuszbuda/brain-segmentation-pytorch/raw/master/assets/TCGA_CS_4944.png)  
On my device(RTX 4090), lotusInfer takes only 0.66 seconds to execute the U-Net model and correctly render out the abnormal organization.   
![abnormal area](records/abnormal_organization.png)

## resnet18 Demo
![German_Shepherd](models/resnet/German_Shepherd.jpg) 
On my device, lotusInfer takes only 0.66 seconds to execute the resnet18 model and correctly figure out the species. 
![resnet18_speedtest](records/Recording_resnet_infer.gif) 





