cmake_minimum_required(VERSION 3.5)

project(mnist LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
#set(InferenceEngine_DIR /opt/intel/openvino_2022/runtime/cmake)
#SET(ENV{PKG_CONFIG_PATH} /opt/intel/openvino_2022/runtime/lib/intel64/pkgconfig)
#set(OpenVINO_DIR "/opt/intel/openvino_2022/runtime") #可在子目录查找
#set(CMAKE_MODULE_PATH  /opt/intel/openvino_2022/runtime/lib/)
add_executable(mnist main.cpp src/yolov5_openvino.cpp)
#include_directories(${CMAKE_SOURCE_DIR}/include/)
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(/opt/intel/openvino_2022/runtime/include)

#find_package(PkgConfig  REQUIRED)
#PKG_SEARCH_MODULE(PKG_OPENVINO REQUIRED openvino)
#link_directories(/opt/intel/openvino_2022/runtime/lib/intel64/)

find_package(OpenCV REQUIRED)
#find_package(InferenceEngine REQUIRED)

#include_directories(/opt/intel/openvino_2022/runtime/include/ie)
#include_directories(/opt/intel/openvino_2022/runtime/include)
target_link_libraries(mnist ${OpenCV_LIBS})

target_link_libraries(mnist /opt/intel/openvino_2022/runtime/lib/intel64/libopenvino.so)



#find_package(OpenVINO REQUIRED)
#add_executable(ov_app main.cpp)
#target_link_libraries(ov_app PRIVATE openvino::runtime)

#add_executable(ov_c_app main.c)
#target_link_libraries(ov_c_app PRIVATE openvino::runtime::c)


