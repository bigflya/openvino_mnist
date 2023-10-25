cmake_minimum_required(VERSION 3.5)

project(mnist LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(mnist src/main.cpp src/yolov5_openvino.cpp)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(/opt/intel/openvino_2022/runtime/include)


find_package(OpenCV REQUIRED)



target_link_libraries(mnist ${OpenCV_LIBS})

target_link_libraries(mnist /opt/intel/openvino_2022/runtime/lib/intel64/libopenvino.so)






