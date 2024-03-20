#include <opencv2/core/utility.hpp>
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include <unistd.h>
#include <sstream>
#include <stdio.h>
// using namespace std;
// using namespace cv;

int main( int argc, const char** argv )
{   
    cv::Mat src = cv::imread("./hexagon.png");
    cv::Mat imageGray;
    cv::cvtColor(src, imageGray, CV_RGB2GRAY);
    
   
    imageGray = ~imageGray;
   

    cv::threshold(imageGray, imageGray, 20, 255, cv::THRESH_BINARY + cv::THRESH_OTSU);
    

    cv::Mat imageThin(imageGray.size(), CV_32FC1);
    cv::distanceTransform(imageGray, imageThin, CV_DIST_L2, 3);  //距离变换
    cv::Mat distShow;
    distShow = cv::Mat::zeros(imageGray.size(), CV_8UC1);
    float maxValue = 0;
    cv::Point Pt(0, 0);
    for (int i = 0; i < imageThin.rows; i++)
    {
        for (int j = 0; j < imageThin.cols; j++)
        {
            distShow.at<uchar>(i, j) = imageThin.at<float>(i, j);
            //把float转换成uchar之后，距离越远的地方越亮
            if (imageThin.at<float>(i, j) > maxValue)
            {
                maxValue = imageThin.at<float>(i, j);  //获取距离变换的最大值
                Pt = cv::Point(j, i);  //最大值的坐标
            }
        }
    }
    cv::normalize(distShow, distShow, 0, 255, CV_MINMAX); //为了显示清晰，做了0~255归一化
    cv::circle(src, Pt, maxValue, cv::Scalar(0, 0, 255), 3);
    cv::circle(src, Pt, 3, cv::Scalar(0, 255, 0), 3);

    cv::namedWindow("src1");
    cv::imshow("src1", src);

    while(1){
        char c = (char)cv::waitKey(0);
        if( c == 27 )
            break;
    }


}