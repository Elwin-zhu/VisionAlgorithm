#include <opencv2/core/utility.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
#include <fstream>
#include <iostream>
#include <sys/stat.h>
#include <unistd.h>
//#include <stdio.h>

using namespace std;

const char* keys =
{
 "{help h||}{@image |stuff.jpg|input image file}"
};


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

void drawHist(cv::Mat &hist, const int* histsize, const cv::Scalar& scl, string name)
{
    // 绘制直方图
    int histH = 500;
    int histW = 600;
    int width = cvRound(histW / histsize[0]);
    cv::Mat histOut;
    cv::Mat histImg(histH, histW, CV_8UC3, cv::Scalar(67, 19, 89));

    //归一化, 把直方图的各级频率（概率）归一化到了0~1区间，0~alph区间了。
    cv::normalize(hist, histOut, 1, 0, cv::NORM_INF, -1, cv::Mat());
    //cv::normalize(hist, hist, 1, 0, cv::NORM_L1, -1, cv::Mat());
    //cv::normalize(hist, hist, 1, 0, cv::NORM_L2, -1, cv::Mat());

    for (int i = 1; i < histOut.rows; i++)
    {
        cv::rectangle(histImg, cv::Point(width * (i - 1), histH - 1),
            cv::Point(width * i - 1, histH - cvRound(histH * histOut.at<float>(i - 1)) - 1),
            scl, 1);
    }
    cv::imshow(name, histImg);
}


int main(int argc, const char** argv)
{
    cv::CommandLineParser parser(argc, argv, keys);
    string filename = parser.get<string>(0);
    std::cout << "filename: " << filename << std::endl;

    cv::Mat img = cv::imread(cv::samples::findFile(filename)); //原图
    if (img.empty()) {
        cout << "打开图片失败" <<endl;
        return -1;
    }
 
    cv::imshow("原始图像", img);
    dumpMat(img, "SourceImg");
 
 
    int channels[1] = {0}; 
    int histSize[1] = {256};
    float range[2] = {0, 256};
    const float* ranges[1] = {range}; // 指定每个通道的取值范围
 
    // 计算B通道的颜色直方图
    cv::Mat bHist;
    channels[0] = 0; 
    cv::calcHist(&img, 1, channels, cv::Mat(), bHist, 1, histSize, ranges);
    // 计算G通道的颜色直方图
    cv::Mat gHist;
    channels[0] = 1; 
    cv::calcHist(&img, 1, channels, cv::Mat(), gHist, 1, histSize, ranges);
    // 计算R通道的颜色直方图
    cv::Mat rHist;
    channels[0] = 2; 
    cv::calcHist(&img, 1, channels, cv::Mat(), rHist, 1, histSize, ranges);
 
    drawHist(bHist, histSize, cv::Scalar(255,0,0), "Blue Histogram");
    drawHist(gHist, histSize, cv::Scalar(0,255,0),"Green Histogram");
    drawHist(rHist, histSize, cv::Scalar(0,0,255),"Red Histogram");

    dumpMats2File(bHist, gHist, rHist, "logBGR.txt");
 
    cv::waitKey(0);
}