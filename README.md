# lotusInfer

## Overview

lotusInfer is a deep learning inference framework in development, supporting CUDA accleration. The framework includes implemented CUDA operators like 2D-convolution, with ongoing development of additional operators such as pooling and expression. The long-term goal is to make this framework powerful enough to support Large Language Model (LLM) inference.

## Development Environment and Libraries Used
- **Operating System**: WSL Ubuntu 22.04.4 
- **C++ Standard**: C++20
- **CUDA Compute Capability**: 8.6 or above
- **PyTorch Neural Network Model Format**: [PNNX](https://github.com/Tencent/ncnn/tree/master/tools/pnnx)
- **Multi-dimensional Array Library**: [xtensor3](https://xtensor.readthedocs.io/en/latest/)
- **Log Library**: [spdlog](https://github.com/gabime/spdlog)


## Install
```bash
$ git clone https://github.com/skystriker1997/lotusInfer.git
$ cd lotusInfer && mkdir build && cd build
$ cmake .. && make -j
```

## TO-DO List
1. Implement activation and pooling operators
2. Complete the compute graph
3. Continue improving and expanding operator support
4. ...




