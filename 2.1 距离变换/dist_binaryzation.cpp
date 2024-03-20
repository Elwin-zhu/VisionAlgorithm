

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/cuda.hpp>
#include <unistd.h>
#include <sstream>

using std::cout; using std::endl;

int main(int argc, char* argv[])
{
    int threshold_mode = 16; //设置二值化的模式
    std::string image_path = "./image21.jpg";
    cv::Mat img = cv::imread(image_path, 1); //读入图像
    cv::namedWindow("example", 1); //创建一个窗口用于显示源图像，1代表窗口适应图像的分辨率进行拉伸。
    cv::namedWindow("threshold", 1); //显示二值化图像
    
    cv::Mat img_g;  
    cv::cvtColor(img, img_g, cv::COLOR_BGR2GRAY); //灰度化源图像，便于我们观察结果

    cv::Mat img_th;

    while (true)
    {
        
        switch (threshold_mode)
        {
            case(cv::THRESH_BINARY):
                cv::threshold(img_g, img_th, 120, 255, cv::THRESH_BINARY);//大于120就被置为255（纯白色），小于120置为0（黑色）
                break;

            case(cv::THRESH_BINARY_INV):
                cv::threshold(img_g, img_th, 120, 255, cv::THRESH_BINARY_INV);//大于120就被置为0（黑色），小于120置为255（白色）
                break;

            case(cv::THRESH_TRUNC):
                cv::threshold(img_g, img_th, 120, 255, cv::THRESH_TRUNC);//大于120就被置为120，小于120保持不变
                break;

            case(cv::THRESH_TOZERO):
                cv::threshold(img_g, img_th, 120, 255, cv::THRESH_TOZERO);//大于120保持不变，小于120置为0（黑色）
                break;

            case(cv::THRESH_TOZERO_INV):
                cv::threshold(img_g, img_th, 120, 255, cv::THRESH_TOZERO_INV);//大于120就被置为0(黑色），小于120保持不变
                break;

            case(cv::THRESH_OTSU):
                cv::threshold(img_g, img_th, 120, 255, cv::THRESH_OTSU);
                //大津阈值，计算得到的自适应阈值将替代120，大于自适应阈值的置为255，小于自适应阈值的置为0
                break;

            case(cv::THRESH_TRIANGLE):
                cv::threshold(img_g, img_th, 120, 255, cv::THRESH_TRIANGLE);
                //与大津阈值类似，计算得到的自适应阈值将替代120，大于自适应阈值的置为255，小于自适应阈值的置为0
                break;

            case(7):
                int adaptiveMethod = 1; //0:cv::ADAPTIVE_THRESH_MEAN_C, 1:ADAPTIVE_THRESH_GAUSSIAN_C 
                int blockSize = 3; //像素领域大小
                double C = 3.0; //从求出的阈值减掉的值，可以抑制噪声
                cv::adaptiveThreshold(img_g, img_th, 255, adaptiveMethod, cv::THRESH_BINARY, blockSize, C);
                //自适应阈值，不像上面针对整个图像的阈值，它的阈值对每个像素点都是变化的，它在一个像素的邻域计算用于该点的阈值。
                break;

            /*default:
                break;*/
        }


        if (img.empty() == false) //图像不为空则显示图像
        {
            cv::imshow("example", img_g);
        }
        if (img_th.empty() == false) //图像不为空则显示图像
        {
            cv::imshow("threshold", img_th);
        }
        
        //sleep(1);
        int  key = cv::waitKey(30); //等待30ms
        //printf("key: %d \n", key);
        threshold_mode = key - 48;
        if (key ==  27) //按下q退出
        {
            break;
        }

    }

    cv::destroyAllWindows(); //关闭所有窗口
        
    return 0;
}

