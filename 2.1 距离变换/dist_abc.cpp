#include <opencv2/core/utility.hpp>
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include <stdio.h>
#include <unistd.h>
#include <sstream>
// using namespace std;
// using namespace cv;

/*
    cv::distanceTransform(imageGray, imageThin, CV_DIST_L2, 3); 
    参数1: 8-bit, 单通道输入图片
    参数2: 输出结果中包含计算的距离，这是一个32-bit float 单通道的Mat类型，大小与输入图片相同
    参数3: distanceType计算距离的类型
            distanceType        maskSize          a \ b \ c     
            CV_DIST_C            3（3X3）         a = 1, b = 1
            CV_DIST_L1           3（3X3）         a = 1, b = 2
            CV_DIST_L2           3（3X3）         a=0.955, b=1.3693
            CV_DIST_L2           5（5X5）         a=1, b=1.4, c=2.1969
    参数4: maskSize – 距离变换掩码矩阵的大小
            3（CV_DIST_L1、 CV_DIST_L2 、CV_DIST_C）
            5（CV_DIST_L2 ）
            CV_DIST_MASK_PRECISE (这个只能在4参数的API中使用)
　　参数5: 目标矩阵的数据类型　　　　　　CV_8U

    说明: 其中 a b c 含义：在这个函数中计算每个非0像素到黑色像素（0值像素）的最短距离，因此需要通过最短的移动方式找到这个点计算他们之间的值。
          通常来说移动有水平方向、竖直方向、对角方向、跳跃式几个移动方法。
          虽然计算距离的方法都是一些很基础的公式，但是这个这个掩码矩阵必须是对阵的，因此掩码矩阵上所有水平和竖直方向的变化量，这里用 a 代表；对角方向的变化量用 b 代表；跳跃移动的变化量用 c 代表。
          CV_DIST_C、CV_DIST_L1、CV_DIST_L2（maskSize=5）的计算结果是精确的，CV_DIST_L2（maskSize=3）是一个快速计算方法
*/

void zoom(cv::Mat& m, cv::Mat& d, int v)
{
    for (int i = 0; i < m.rows; i++)
    {
        for (int j = 0; j < m.cols; j++)
        {
            if (m.at<float>(i, j) > v)
            {
                d.at<uchar>(i, j) = 255;   //符合距离大于最大值一定比例条件的点设为255
            }
        }
    }
}

int main( int argc, const char** argv )
{
    float maxValue = 0;  //保存距离变换矩阵中的最大值
    int   maxValueInt = 5;
    cv::Mat src = cv::imread("./abc.png", 0);

    cv::namedWindow("image");
    cv::imshow("image", src);

    cv::imwrite("./abc.bmp", src);
    cv::Mat imageGray=~src;  //取反, 255减去Mat中的每个元素的值，得到取反后的Mat中对应的取反值。
    cv::imwrite("./abc_N.bmp", imageGray);

    cv::GaussianBlur(imageGray, imageGray, cv::Size(5, 5), 2); //滤波-去除杂点
    cv::threshold(imageGray, imageGray, 10, 255, cv::THRESH_BINARY);//大于10就被置为255（纯白色），小于10置为0（黑色）
    cv::imwrite("./abc_N_Thr.bmp", imageGray);

    cv::namedWindow("imageGray");
    cv::imshow("imageGray",imageGray);
    cv::Mat imageThin(imageGray.size(), CV_32FC1);
    cv::distanceTransform(imageGray, imageThin, CV_DIST_L2, 3);  //距离计算，imageGray矩阵的子集S是字母“ABCD”之外的背景，因为背景是纯黑色，是0，而不是“ABCD”纯白色的字母
    cv::imwrite("./abc_N_Thr_Trans.bmp", imageThin);   //输出图片是F(p)矩阵，

    cv::Mat distShow;
    distShow = cv::Mat::zeros(imageGray.size(), CV_8UC1); 

    for (int i = 0; i < imageThin.rows; i++)
    {
        for (int j = 0; j < imageThin.cols; j++)
        {
            if (imageThin.at<float>(i, j) > maxValue)
            {
                maxValue = imageThin.at<float>(i, j);  //获取距离变换的最大值
            }
        }
    }
    printf("maxValue: %f\n", maxValue);

    for (int i = 0; i < imageThin.rows; i++)
    {
        for (int j = 0; j < imageThin.cols; j++)
        {
            // if (imageThin.at<float>(i, j) > maxValue / 1.9)
            if (imageThin.at<float>(i, j) > maxValueInt)
            {
                distShow.at<uchar>(i, j) = 255;   //符合距离大于最大值一定比例条件的点设为255
            }
        }
    }
    cv::namedWindow("distShow");
    cv::imshow("distShow", distShow);
    // cv::startWindowThread();//开始不断的更新图片

    while(1){
        char c = (char)cv::waitKey(0); //等待30ms
        // cv::destroyAllWindows();
        if(c == 82)
            maxValueInt++;
        else if(c == 84)
            maxValueInt--;
        // printf("%d ", maxValueInt);
        // fflush(stdout); 
        distShow.setTo(cv::Scalar(0));
        zoom(imageThin, distShow, maxValueInt);
        cv::imshow("distShow", distShow);
        if( c == 27 )
            break;
    }
}