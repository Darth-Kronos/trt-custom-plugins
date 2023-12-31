# trt-custom-plugins
This repository contains custom TensorRT plugins for specialized operators. These plugins can be seamlessly integrated into your TensorRT workflow to enhance the capabilities of your deep learning models.

## Prerequisites
To build the TensorRT-OSS components, you will first need the following software packages.

* [CUDA](https://developer.nvidia.com/cuda-toolkit)
  * Recommended versions:
  * cuda-12.2.0 + cuDNN-8.8
  * cuda-11.8.0 + cuDNN-8.8
* [GNU make](https://ftp.gnu.org/gnu/make/) >= v4.1
* [cmake](https://github.com/Kitware/CMake/releases) >= v3.13
* [python](<https://www.python.org/downloads/>) >= v3.6.9, <= v3.10.x
* [pip](https://pypi.org/project/pip/#history) >= v19.0
* Essential utilities
  * [git](https://git-scm.com/downloads), [pkg-config](https://www.freedesktop.org/wiki/Software/pkg-config/), [wget](https://www.gnu.org/software/wget/faq.html#download)
* [PyTorch](https://pytorch.org/get-started/locally/) >= 2.0

## Setup
Clone the repository and submodules
```bash
git clone https://github.com/Darth-Kronos/trt-custom-plugins
cd trt-custom-plugins
git submodule update --init --recursive
```
## Usage
To use the custom operators in your TensorRT project, ensure that the prerequisites are met and follow these steps:

1. Copy the required plugin from `plugin` folder to `TensorRT/plugin`.
    ```bash
    cp -r plugin/<plugin>/trtPlugin TensorRT/plugin/<plugin>
    ```
    For more details on the folder structure of [plugin](plugin) check [Plugin Readme](plugin/README.md).

2. Add the `plugin` to [TensorRT's Plugin CMake](TensorRT/plugin/CMakeLists.txt)
    ```cmake
    set(PLUGIN_LISTS
        ...
        ...
        <plugin>
    )
    ```
3. Building TensorRT
* Exporting TensorRT's path
    ```bash
    export TRT_LIBPATH=`pwd`/TensorRT
    ```
* Generate Makefiles and build.

    **Example: Linux (x86-64) build with  cuda-11.8**
	```bash
	cd Tensort
	mkdir -p build && cd build
	cmake .. -DTRT_LIB_DIR=$TRT_LIBPATH -DTRT_OUT_DIR=`pwd`/out -DCUDA_VERSION=11.8
	make -j$(nproc)
	``` 
    For more information follow [TensorRT's README](https://github.com/NVIDIA/TensorRT/)
4. Exporting and converting
* Building Torchscript operator
    ```bash
    cd plugin/<plugin>/tsPlugin
    mkdir -p build && cd build
    cmake -DCMAKE_PREFIX_PATH="$(python3 -c 'import torch.utils; print(torch.utils.cmake_prefix_path)')" ..
    make -j
    cd ../../..
    ```
* Exporting an example model
    ```bash
    cd plugin/<plugin>
    python3 export.py
    cd ../..
    ```
* Convert the model using the previously built `trtexec`
    ```bash
    TensorRT/build/out/trtexec --onnx=plugin/<plugin>/model.onnx --saveEngine=plugin/<plugin>/model.engine --plugins=TensorRT/build/out/libnvinfer_plugin.so
    ```

## Contents

| Plugin | Name | Versions |
|---|---|---|
| [cosLUPlugin](plugin/cosLUPlugin) | CosLUPlugin_TRT | 0 |

## Disclamer
- Raise issue if you need a custom plugin for trt
