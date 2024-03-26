#include <opencv2/core/utility.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;

void drawHist(cv::Mat &hist, const int* histsize, string name)
{
    // 绘制直方图
    int histH = 500;
    int histW = 600;
    int width = cvRound(histW / histsize[0]);
    cv::Mat histImg(histH, histW, CV_8UC1, cv::Scalar(0, 0, 0));

    //归一化
    cv::normalize(hist, hist, 1, 0, cv::NORM_INF, -1, cv::Mat());
    //cv::normalize(hist, hist, 1, 0, cv::NORM_L1, -1, cv::Mat());
    //cv::normalize(hist, hist, 1, 0, cv::NORM_L2, -1, cv::Mat());

    for (int i = 1; i < hist.rows; i++)
    {
        cv::rectangle(histImg, cv::Point(width * (i - 1), histH - 1),
            cv::Point(width * i - 1, histH - cvRound(histH * hist.at<float>(i - 1)) - 1),
            cv::Scalar(255, 255, 255), -1);
    }
    cv::imshow(name, histImg);
}

const char* keys =
{
 "{help h||}{@image |stuff.jpg|input image file}"
};

int main(int argc, const char** argv)
{
    cv::CommandLineParser parser(argc, argv, keys);
    string filename = parser.get<string>(0);

    cv::Mat image = cv::imread(cv::samples::findFile(filename));
    if (image.empty()) {
        cout << "打开图片失败" <<endl;
        return -1;
    }
    cv::Mat gray, equImg;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    //直方图均衡化的主要作用是提高图片的饱和度，即让直方图尽量扩散分布在0-255的亮度区间，不要集中在某一个区间。注：直方图均衡化只能在单通道图片进行
    cv::equalizeHist(gray, equImg);

    //定义直方图参数
    cv::Mat hist1, hist2;  // 存放直方图结果
    const int channels[] = { 0 };  // 通道索引
    const int histsize[] = { 256 }; // 直方图的维度，即像素最小值和最大值的差距
    float inrange[] = { 0,255 };
    const float* ranges[] = {inrange};  // 像素灰度值范围

    cv::calcHist(&gray, 1, channels, cv::Mat(), hist1, 1, histsize, ranges);
    cv::calcHist(&equImg, 1, channels, cv::Mat(), hist2, 1, histsize, ranges);

    drawHist(hist1, histsize, "Source-Histogram");
    drawHist(hist2, histsize, "Equalized-Histogram");
    
    cv::imshow("Source-Gray", gray);
    cv::imshow("Equalized", equImg);
    cv::waitKey(0);
    cv::destroyAllWindows();

    return 0;
}
