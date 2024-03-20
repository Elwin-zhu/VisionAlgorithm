#指定需要的cmake的最低版本
cmake_minimum_required(VERSION 2.8)
#创建工程
project(OpenCVTest)
#指定C++语言为C++ 11
set(CMAKE_CXX_FLAGS "-std=c++11")

# # 设置编译器路径
# set(CMAKE_C_COMPILER "/home/elwin/sdb1/halley6/buildroot/buildroot/output/host/bin/mips-linux-gnu-gcc")
# set(CMAKE_CXX_COMPILER "/home/elwin/sdb1/halley6/buildroot/buildroot/output/host/bin/mips-linux-gnu-g++")
# # 设置目标平台和架构
# set(CMAKE_SYSTEM_NAME Linux)
# set(CMAKE_SYSTEM_PROCESSOR mips)

#查找OpenCV 4.5.2安装路径
find_package(OpenCV 3.4.9 REQUIRED)
#引入OpenCV头文件路径
include_directories(${OpenCV_INCLUDE_DIRS}) 
#指定编译OpenCVTestDemo.cpp程序编译后生成可执行文件OpenCVTestDemo
add_executable(dist_byVector dist_byVector.cpp)
#指定可执行文件OpenCVTestDemo链接OpenCV lib
target_link_libraries(dist_byVector ${OpenCV_LIBS})
message(STATUS "result:${OpenCV_LIBS}")
