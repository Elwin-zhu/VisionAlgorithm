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
    m.step = m.cols*m.channel()*B （B是字节数，每个通道channel所占的字节数）
    m.step代表以字节为单位的图像的行宽，即m矩阵中每一行所占的字节数(包括填补像素)，因为每一行中的一个像素可以是多个channel通道，即使你的图像元素类型不是uchar，step仍然带代表着行的字节数
    比如：一幅长400高500像素的彩色图片，导入到cv::Mat后，每一行400个单元，每个单元有3个channel，每个channel站1字节，则Mat的一行是step：400*3=1200字节
*/
void dumpMat(cv::Mat& m, const std::string& tag)
{
    int r = 0;
    std::cout << "\n" << tag << "*************************************************" << std::endl;
    std::cout << "rows: "<< m.rows << ", cols: " << m.cols << ", dims: " << m.dims << ", channels: " << m.channels() << ", step: " << m.step << std::endl;
    std::cout << "data: " << std::endl;

    for (int i = 0; i < m.rows; ++i) {
        for (int j = 0; j < m.cols; ++j) {
            cv::Vec3b pixel = m.at<cv::Vec3b>(i, j);
            std::cout << "(" << static_cast<int>(pixel[0]) << ", "
                      << static_cast<int>(pixel[1]) << ", "
                      << static_cast<int>(pixel[2]) << ") " << "\t" << r++;
        }
        std::cout << std::endl;
    }

    std::cout << std::endl;
}

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

/*
   以下输出特别注意：m.step = m.cols*m.channel()*B （B是字节数，每个通道channel所占的字节数）
*/
    std::cout << "\tm1\t" << "m2\t" << "m3\t" << std::endl;
    std::cout << "rows: " << "\t" << m1.rows << "\t" << m2.rows << "\t"<< m3.rows << std::endl;
    std::cout << "cols: " << "\t" << m1.cols << "\t" << m2.cols << "\t"<< m3.cols << std::endl;
    std::cout << "dims: " << "\t" << m1.dims << "\t" << m2.dims << "\t"<< m3.dims << std::endl;
    std::cout << "chns: " << "\t" << m1.channels() << "\t" << m2.channels() << "\t"<< m3.channels() << std::endl;
    std::cout << "step: " << "\t" << m1.step << "\t" << m2.step << "\t"<< m3.step << std::endl;
    std::cout << "\ndata: " << std::endl;
    std::cout << "Range" << "\tm1.data\t\t  " << "m2.data\t\t" << "m3.data\t" << std::endl;

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

void drawHist(cv::Mat &hist, const int* histsize, string name)
{
    // 绘制直方图
    int histH = 500;
    int histW = 600;
    int width = cvRound(histW / histsize[0]);
    cv::Mat hist_out;
    cv::Mat histImg(histH, histW, CV_8UC1, cv::Scalar(0, 0, 0));

    // dumpMat(hist, "Histogram+");
    //归一化, 特别注意，此时alph为1时，0.xxxx的小数了，特别注意此时的小数转整数，以及画直方图时的坐标转换
    cv::normalize(hist, hist_out, 1, 0, cv::NORM_INF, -1, cv::Mat());
    //cv::normalize(hist, hist, 1, 0, cv::NORM_L1, -1, cv::Mat());
    //cv::normalize(hist, hist, 1, 0, cv::NORM_L2, -1, cv::Mat());
    // dumpMat(hist, "Histogram-");
    // dumpMats2File(hist, hist_out, hist_out, std::to_string(1));
    dumpMats2File(hist, hist_out, hist_out, "log2.txt");

    for(int i = 1; i < hist_out.rows; i++)
    {
        printf("%d, ", cvRound(histH * hist_out.at<float>(i - 1)));
        fflush(stdout);
        cv::rectangle(histImg, cv::Point(width * (i - 1), histH - 1),
            cv::Point(width * i - 1, histH - cvRound(histH * hist_out.at<float>(i - 1)) - 1),
            cv::Scalar(255, 255, 255), -1);
    }
    std::cout << std::endl;
    cv::imshow(name, histImg);
}

const char* keys =
{
 "{help h||}{@image |stuff.jpg|input image file}"
 "{help h||}{@image |stuff.jpg|input image file}"
};

int main(int argc, const char** argv)
{
    cv::CommandLineParser parser(argc, argv, keys);
    string filename1 = parser.get<string>(0);
    std::cout << "filename1: " << filename1 << std::endl;
    string filename2 = parser.get<string>(1);
    std::cout << "filename2: " << filename2 << std::endl;

    cv::Mat image1 = cv::imread(cv::samples::findFile(filename1)); //原图
    cv::Mat image2 = cv::imread(cv::samples::findFile(filename2)); //模板图
    if (image1.empty() || image2.empty()) {
        cout << "打开图片失败" <<endl;
        return -1;
    }
    // dumpMats2File(image1, image2, image1, "log.txt");
    // dumpMat(image1, "Image");
    cv::Mat gray1, gray2, equImg;
    cv::cvtColor(image1, gray1, cv::COLOR_BGR2GRAY);
    cv::cvtColor(image2, gray2, cv::COLOR_BGR2GRAY);
    // dumpMat(gray1, "ImageGray");
    cv::equalizeHist(gray1, equImg);

    //定义直方图参数
    cv::Mat hist1, hist2, hist3;  // 存放直方图结果
    const int channels[] = { 0 };  // 通道索引
    const int histsize[] = { 256 }; // 直方图的维度，即像素最小值和最大值的差距
    float inrange[] = { 0,255 };
    const float* ranges[] = {inrange};  // 像素灰度值范围

    // 计算两张图像直方图
    //cv::calcHist(&image1, 1, channels, cv::Mat(), hist1, 1, histsize, ranges);
    //cv::calcHist(&image2, 1, channels, cv::Mat(), hist2, 1, histsize, ranges);
    cv::calcHist(&gray1, 1, channels, cv::Mat(), hist1, 1, histsize, ranges);
    cv::calcHist(&gray2, 1, channels, cv::Mat(), hist2, 1, histsize, ranges);

    cv::calcHist(&equImg, 1, channels, cv::Mat(), hist3, 1, histsize, ranges);

    // 归一化
    drawHist(hist1, histsize, "原图直方图");
    drawHist(hist2, histsize, "模板直方图");
    drawHist(hist3, histsize, "均值化后直方图");

    //1.计算两张图像直方图的累积概率
    float hist1_cdf[256] = { hist1.at<float>(0) };
    float hist2_cdf[256] = { hist2.at<float>(0) };
    for (int i = 1; i < 256; i++)
    {
        hist1_cdf[i] = hist1_cdf[i - 1] + hist1.at<float>(i);
        hist2_cdf[i] = hist2_cdf[i - 1] + hist2.at<float>(i);
    }
    //2.构建累积概率误差矩阵
    float diff_cdf[256][256];
    for (int i = 0; i < 256; i++)
    {
        for (int j = 0; j < 256; j++)
        {
            //fabs函数是一个求绝对值的函数，求出x的绝对值，和数学上的概念相同，函数原型是extern float fabs(float x)，用法是#include <math.h>。
            diff_cdf[i][j] = fabs(hist1_cdf[i] - hist2_cdf[j]);
        }
    }
    //3.生成LUT映射表，LUT: Look Up Table, 查找表，也称色彩表、色彩图、，索引记录和调色板的连接。
    cv::Mat lut(1, 256, CV_8U);
    for (int i = 0; i < 256; i++)
    {
        // 查找源灰度级为i的映射灰度
        //　和i的累积概率差值最小的规定化灰度
        float min = diff_cdf[i][0];
        int index = 0;
        //寻找累积概率误差矩阵中每一行中的最小值
        for (int j = 1; j < 256; j++)
        {
            if (min > diff_cdf[i][j])
            {
                 min = diff_cdf[i][j];
                 index = j;
            }
        }
        lut.at<uchar>(i) = (uchar)index;
    }

    cv::Mat matchImg;
    //cv::LUT(image1, lut, matchImg);
    cv::LUT(gray1, lut, matchImg);
    
    //cv::imshow("原图", image1);
    //cv::imshow("模板图", image2);
    cv::imshow("原图", gray1);
    cv::imshow("模板图", gray2);
    cv::imshow("匹配图", matchImg);
    cv::imshow("equ", equImg);

    cv::Mat hist4;
    cv::calcHist(&matchImg, 1, channels, cv::Mat(), hist4, 1, histsize, ranges);
    drawHist(hist4, histsize, "匹配直方图");

    cv::waitKey(0);
    cv::destroyAllWindows();

    return 0;
}
