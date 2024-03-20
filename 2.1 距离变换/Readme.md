图像距离变换相关demo，帮助了解距离变换的本质，每个文件需要手动单独编译出目标文件进行学习和测试，在ubuntu20.04系统下编译和运行OK。
每个文件的编译方法：
1、在CMakeLists.txt中替换要编译的cpp文件，比如：在add_executable(dist_byVector dist_byVector.cpp)中要编译dist_byVector.cpp文件，且编译的目标执行文件为dist_byVector。
2、指定可执行文件的名称，并与opencv库link，比如：在target_link_libraries(dist_byVector ${OpenCV_LIBS})中，dist_byVector为目标执行文件。
3、在当前CMakeLists.txt文件夹中，在终端中运行：cmake . (注意后面有一个点，表示当前目录)，会生产Makefile文件。
4、执行make 或者 make clean
5、运行编译出来的demo：./dist_byVector
