#include <opencv2/core/utility.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
#include <fstream>
#include <iostream>
#include <sys/stat.h>
#include <unistd.h>
//#include <stdio.h>

using namespace std;
/*
  在这个例子中，我们首先定义了直方图的大小和范围，然后使用calcHist函数计算了图像的直方图。
  之后，我们使用normalize函数对直方图进行了归一化，以便我们可以将其绘制到一个图像上。
  最后，我们使用line函数在一个空白的图像上绘制了直方图的线条。
*/
 
void dumpMats2File(cv::Mat& m1, cv::Mat& m2, cv::Mat& m3, const std::string& fileName)
{
    std::ofstream of;
    std::streambuf* oldbuf;
    int r = 0;

    if(fileName.compare("1") != 0){
        if(access(fileName.c_str(), F_OK) != -1){
            std::cout << "remove " << fileName << std::endl;
            remove(fileName.c_str());
        }
        of.open(fileName);
        if(!of.is_open()){
            std::cerr << "Open file error!!" << std::endl;
            return;
        }
        //备份的stdout buffer
        oldbuf = std::cout.rdbuf();
        //将cout重定向到of
        std::cout.rdbuf(of.rdbuf());
    }

    std::cout << "\t\tm1\t" << "m2\t" << "m3\t" << std::endl;
    std::cout << "rows: " << "\t" << m1.rows << "\t" << m2.rows << "\t"<< m3.rows << std::endl;
    std::cout << "cols: " << "\t" << m1.cols << "\t" << m2.cols << "\t"<< m3.cols << std::endl;
    std::cout << "dims: " << "\t" << m1.dims << "\t" << m2.dims << "\t"<< m3.dims << std::endl;
    std::cout << "chns: " << "\t" << m1.channels() << "\t" << m2.channels() << "\t"<< m3.channels() << std::endl;
    std::cout << "step: " << "\t" << m1.step << "\t" << m2.step << "\t"<< m3.step << std::endl;
    std::cout << "\ndata: " << std::endl;
    std::cout << "Range" << "\tm1.data\t\t" << "m2.data\t\t" << "m3.data\t" << std::endl;

#if 0 //整数会把float变成1，根据step是4字节float。
    for (int i = 0; i < m1.rows; ++i) {
        for (int j = 0; j < m1.cols; ++j) {
            int p0 = cvRound(m1.at<float>(i));
            int p1 = cvRound(m2.at<float>(i));
            int p2 = cvRound(m3.at<float>(i));
            std::cout << r++ << "\t" << setw(6) << p0 << "\t\t" << setw(6) << p1 << "\t\t" << setw(6) << p2 << "\t";
        }
        std::cout << std::endl;
    }
#else
    for (int i = 0; i < m1.rows; ++i) {
        for (int j = 0; j < m1.cols; ++j) {
            float p0 = m1.at<float>(i);
            float p1 = m2.at<float>(i);
            float p2 = m3.at<float>(i);
            std::cout << r++ << "\t" << setw(8) << p0 << "\t\t" << setw(8) << p1 << "\t\t" << setw(8) << p2 << "\t";
        }
        std::cout << std::endl;
    }
#endif

    std::cout << std::endl;

    if(fileName.compare("1") != 0){
        //恢复原来的stdout buffer
        std::cout.rdbuf(oldbuf);
    }
    of.close();
}

int main() 
{
    // 读取图像
    cv::Mat image = cv::imread("test1.jpg", cv::IMREAD_GRAYSCALE);
    if (image.empty()) {
        std::cout << "图像加载失败" << std::endl;
        return -1;
    }
 
    // 初始化直方图参数
    int histSize = 256; // bin的个数
    float range[] = {0, 256}; // 灰度级的范围
    const float* histRange = {range};
    cv::Mat hist, hist_out;
    bool uniform = true, accumulate = false;
 
    /*
    计算直方图
    void cv::calcHist(const Mat* images,  // 图像或图像集合，集合内所有的图像应具有相同的尺寸和数据类型，并且数据类型只能是CV_8U、CV_16U和CV_32F三种中的一种，但是不同图像的通道数可以不同。
				int nimages,   		  // 输入图像的数量（当处理多幅图像时使用）
				const int* channels,  // 需要统计的通道索引数组，第一个图像的通道索引从0到images[0].channels()-1（灰度图设置为[0]），第二个图像通道索引从images[0].channels()到images[0].channels()+ images[1].channels()-1，以此类推
				InputArray mask,   // 可选的操作掩码，如果是空矩阵则表示图像中所有位置的像素都计入直方图中，如果矩阵不为空，则必须与输入图像尺寸相同且数据类型为CV_8U
				OutputArray hist,  // 输出的统计直方图结果，是一个dims维度的数组,cv::Mat形式
 				int dims,  // 直方图的维数。对于灰度图像，默认为1；对于彩色图像，默认为3（每个颜色通道一个维度）
 				const int* histSize,  // 存放每个维度直方图的数组的尺寸，即像素最小值和最小值的差距
 				const float** ranges,  // 每个图像通道中灰度值的取值范围
 				bool uniform = true,   // 直方图是否均匀的标志符，默认状态下为均匀
 				bool accumulate = false  // 是否累积统计直方图的标志,如果累积（true），则统计新图像的直方图时之前图像的统计结果不会被清除，该功能主要用于统计多个图像整体的直方图
 				);

    */
    cv::calcHist(&image, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange, uniform, accumulate);

    dumpMats2File(hist, hist, hist, "test1jpg.log");

    // printf("%d, %d, %d, %d\n", hist.rows, hist.cols, hist.dims, (int)hist.step);
    // printf("data: %02X, %02X, %02X, %02X, %02X, %02X\n", hist.data[0], hist.data[1], hist.data[2], hist.data[3], hist.data[4], hist.data[5]);
    // fflush(stdout);
 
    // 正规化直方图
    int hist_w = 512, hist_h = 400;
    int bin_w = cvRound((double)hist_w/histSize); //cvRound: 对一个double型的数进行四舍五入，并返回一个整型数！
    cv::Mat histImg(hist_h, hist_w, CV_8UC3, cv::Scalar(0,0,0)); //其实是hist_h*(hist_w*3)的矩阵，因为每个元素有3个通道，CV_8UC3表示是8bit无符号3通道
    // OpenCV的函数normalize()的两个作用:调整矩阵的值范围(归一化处理)、规范化矩阵的范数为某个值，究竟函数normalize()发挥什么作用，这取决于参数norm_type取什么值
    normalize(hist, hist_out, 0, hist_h, cv::NORM_MINMAX, -1, cv::Mat());
    // dumpMats2File(hist, hist_out, hist_out, std::to_string(1));
    dumpMats2File(hist, hist_out, hist_out, "log.txt");
   
    // 绘制直方图
    for(int i = 1; i < histSize; i++) {
        printf("%d, ", cvRound(hist_out.at<float>(i)));
        fflush(stdout);
        cv::line(histImg, cv::Point(bin_w*(i-1), hist_h - cvRound(hist_out.at<float>(i-1))),
                 cv::Point(bin_w*(i), hist_h - cvRound(hist_out.at<float>(i))),
                 cv::Scalar(0, 0, 255), 2, 8, 0);
    }

    // fflush(stdout);
    // printf("\n");
    //均衡化原图
    cv::Mat equ;
    cv::equalizeHist(image, equ);
 
 
    // 显示原图
    cv::imshow("Source", image);
    // 显示直方图
    cv::imshow("Histogram", histImg);
    //显示均衡后的图
    cv::imshow("Equalized", equ);
    cv::waitKey(0);
    std::cout << std::endl;
    return 0;
}