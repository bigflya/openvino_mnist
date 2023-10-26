# openvino_mnist

## 仓库介绍 ：
openvino_mnist是一个关于openvino的一个推理项目，这里给出两套代码分别是推理视频的，和推理图片的。推理过程均在Ubuntu20.04环境下，通过cmake编译构建项目，且采用最新的openvino LTS版本（截至2023/10/26）。供大家学习使用。这里也给出Ubuntu20安装openvino的教程[如何在Ubuntu上安装openvino runtime运行环境](https://bigflya.top/2023/06/03/openvino%20%E5%9C%A8ubuntu20%E4%B8%8A%E7%9A%84%E5%AE%89%E8%A3%85%E9%83%A8%E7%BD%B2/)



## 快速使用教程

克隆此仓库到本地，建议先看[如何在Ubuntu上安装openvino runtime运行环境](https://bigflya.top/2023/06/03/openvino%20%E5%9C%A8ubuntu20%E4%B8%8A%E7%9A%84%E5%AE%89%E8%A3%85%E9%83%A8%E7%BD%B2/)，按照步骤完成相关环境的搭建。
图片检测











## 模型的数据集下载

链接：https://pan.baidu.com/s/15k1x9COE3zdV-b7wLWKOWg 
提取码：bfly


## 工程结构

这里给出picture_mnist_Detect_Project的目录结构，video_mnist_Detect_Project的目录结构与此相同。

```txt

picture_mnist_Detect_Project
├── build
│   ├── CMakeCache.txt
│   ├── CMakeFiles
│   │   ├── 3.16.3
│   │   │   ├── CMakeCXXCompiler.cmake
│   │   │   ├── CMakeDetermineCompilerABI_CXX.bin
│   │   │   ├── CMakeSystem.cmake
│   │   │   └── CompilerIdCXX
│   │   │       ├── a.out
│   │   │       ├── CMakeCXXCompilerId.cpp
│   │   │       └── tmp
│   │   ├── cmake.check_cache
│   │   ├── CMakeDirectoryInformation.cmake
│   │   ├── CMakeOutput.log
│   │   ├── CMakeTmp
│   │   ├── Makefile2
│   │   ├── Makefile.cmake
│   │   ├── mnist.dir
│   │   │   ├── build.make
│   │   │   ├── cmake_clean.cmake
│   │   │   ├── CXX.includecache
│   │   │   ├── DependInfo.cmake
│   │   │   ├── depend.internal
│   │   │   ├── depend.make
│   │   │   ├── flags.make
│   │   │   ├── link.txt
│   │   │   ├── progress.make
│   │   │   └── src
│   │   │       ├── main.cpp.o
│   │   │       └── yolov5_openvino.cpp.o
│   │   ├── progress.marks
│   │   └── TargetDirectories.txt
│   ├── cmake_install.cmake
│   ├── Makefile
│   └── mnist
├── CMakeLists.txt
├── configFiles
│   ├── best1.onnx
│   ├── best2.onnx
│   ├── best.onnx
│   └── classes.txt
├── include
│   └── yolov5_openvino.h
├── save_data
│   ├── result.csv
│   ├── save_1.jpg
│   ├── save_2.jpg
│   ├── save_32.jpg
│   ├── save_3.jpg
│   ├── save_4.jpg
│   ├── save_5.jpg
│   ├── save_7.jpg
│   ├── save_8.jpg
│   ├── save_9.jpg
│   └── save_save.jpg
├── src
│   ├── main.cpp
│   └── yolov5_openvino.cpp
└── test_data
    ├── 1.jpg
    ├── 2.jpg
    ├── 32.jpg
    ├── 3.jpg
    ├── 4.jpg
    ├── 5.jpg
    ├── 7.jpg
    ├── 8.jpg
    ├── 9.jpg
    └── save.jpg

```

./build ：是构建文件夹，大家下载后可以清空build 文件夹下的内容，然后在build文件夹内执行以下命令构建工程：

```shell 
cmake ..
make
```

> 注意： 执行上述命令前需要在main.cpp中修改 proj_dir 对应的地址为自己对应的

./configFiles ： 文件夹下存放模型文件和模型的分类文件

./test_data ：  文件夹下存放测试图片

./save_data ：  文件夹下存放的是运行程序后生成的结果文件

./include ： 文件夹下存放的是一些头文件

./src ： 文件夹下存放的是cpp文件























### 克隆仓库出现错误解决办法

fatal: unable to access 'https://github.com/bigflya/openvino_mnist.git/': Failed to connect to github.com port 443: Connection refused


1  win+r ----> cmd ----->  ipconfig/flushdns


