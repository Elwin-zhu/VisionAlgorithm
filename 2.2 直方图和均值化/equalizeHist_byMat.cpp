#include <opencv2/core/utility.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
#include <fstream>
#include <iostream>
#include <sys/stat.h>
#include <unistd.h>

/*
    m.step = m.cols*m.channel()*B （B是字节数，每个通道channel所占的字节数）
    m.step代表以字节为单位的图像的行宽，即m矩阵中每一行所占的字节数(包括填补像素)，因为每一行中的一个像素可以是多个channel通道，即使你的图像元素类型不是uchar，step仍然带代表着行的字节数
    比如：一幅长400高500像素的彩色图片，导入到cv::Mat后，每一行400个单元，每个单元有3个channel，每个channel站1字节，则Mat的一行是step：400*3=1200字节
*/
void dumpMat(cv::Mat& m, const std::string& tag)
{
    int B = 0;
    B = m.step/(m.cols*m.channels());
    std::cout << tag << " Mat: " << std::endl;
    std::cout << "rows: "<< m.rows << ", cols: " << m.cols << ", dims: " << m.dims << ", channels: " << m.channels() << ", step: " << m.step << ", B: " << B << std::endl;
}

void myCalcHist_Equalize(cv::Mat& m, std::vector<int>& histOut, int* range, cv::Mat& outM, int equType)
{
    int height = m.rows;
    int width = m.cols;
    float hw = height*width;
    int rangeStart = range[0];
    int rangeStop = range[1];
    std::vector<float> p_vec(range[1], 0.0); //概率普
    std::vector<float> c_vec(range[1], 0.0); //累计概率普

    histOut.resize(range[1], 0); //预分配空间，然后初始值为0

    std::cout << "equType: " << equType << std::endl;

    //第1步：统计单通道图片的各灰度级的直方图在histOut中，且计算灰度级在整幅图片中出现的概率p_vec。
    for(int i=0; i<height; i++){
        for(int j=0; j<width; j++){
            histOut[m.at<uchar>(i, j)]++;
        }
    }

    std::cout << "\n range: " << range[1] << std::endl;
    for(int i=0; i<range[1]; i++){
        p_vec[i] = (float)(histOut[i] / hw);
        // printf("%d: %f, ", i, p_vec[i]);
    }

    //第2步：计算累计直方图c_vec, 给后面的NORM_MINMAX均衡用
    for(int i=0; i<range[1]; i++){
        for(int j=0; j<=i; j++){
            c_vec[i] += p_vec[j];
        }
    }

    // std::cout << "\n c_vec: " << std::endl;
    // for(const auto& e : c_vec){
    //     printf("%f, ", e);
    // }
    // fflush(stdout);

    //第3步：均值化原图，根据equType类型来均值化原图
    //第3.1步：先归一化

    float L1_norm = 0.0;
    float L2_norm = 0.0;
    float INF_norm = 0.0;
    float MINMAX_norm = 0.0; 
    double max=0, min=0;
    double fstep = max - min; 
    float alph = 0;
    cv::Point pMax, pMin;
	// outM.create(height, width, CV_8UC1);
    outM= cv::Mat::zeros(height, width, CV_8UC1);
    switch (equType)
    {
    case cv::NORM_L1 :
        // 计算直方图L1_vec, 给后面的NORM_L1均衡用，该效果不明显
        for(int i=0; i<range[1]; i++){
            L1_norm += abs(p_vec[i]);
        }
        printf("L1_norm: %f\n", L1_norm);
        fflush(stdout);
        alph =255; // 灰度图像的最高亮度级别为255
        for(int i=0; i<height; i++){
            for(int j=0; j<width; j++){
                outM.at<uchar>(i, j) = cvRound(histOut[m.at<uchar>(i,j)]*alph/L1_norm);
            }
        }
        // normalize(m, outM, 255, 0, cv::NORM_L1);
        break;
    case cv::NORM_L2 :
        normalize(m, outM, 255, 0, cv::NORM_L2);
        break;
    case cv::NORM_INF :
        normalize(m, outM, 255, 0, cv::NORM_INF);
        break;
    case cv::NORM_MINMAX :
#if 0 //根据图片最大和最小级数来均衡化图片，效果不明显
        cv::minMaxLoc(m, &min, &max, &pMin, &pMax);
#else //根据单通道1 uchar级数来均衡化图片，效果明显
        min = 0;  //单通道灰度图片最小级数为0
        max = 255; //单通道灰度图片最大级数为255
#endif
        fstep = max - min; 
        printf("%f, %f, %f \n", min, max, fstep);
        fflush(stdout);
        for(int i=0; i<height; i++){
            for(int j=0; j<width; j++){
                outM.at<uchar>(i, j) = cvRound(c_vec[m.at<uchar>(i, j)]*fstep + min);
            }
        }
        break;
    default:
        break;
    }


}

void drawHist(cv::Mat &hist, const int* histsize, const cv::Scalar& scl, const std::string& name)
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
    std::stringstream ss;
    std::map<int, std::string> parameters;
    std::cout << "argc: " << argc << std::endl;
    for(int i=1; i<= argc; i++){
        ss << argv[i];
        //std::cout << ss.str() << std::endl;
        parameters[i] = ss.str();
        ss.str("");
        ss.clear();
    }
    std::cout << "NORM_L1: " << cv::NORM_L1 << std::endl;
    std::cout << "NORM_L2: " << cv::NORM_L2 << std::endl;
    std::cout << "NORM_INF: " << cv::NORM_INF << std::endl;
    std::cout << "NORM_MINMAX: " << cv::NORM_MINMAX << std::endl;
    cv::Mat image = cv::imread(cv::samples::findFile(parameters[1]), cv::IMREAD_GRAYSCALE); //原图
    if (image.empty()) {
        std::cout << "打开图片失败" <<std::endl;
        return -1;
    }
    std::cout << "Open image " << parameters[1] << " successful." << std::endl;
    dumpMat(image, "imageSource");
    std::vector<int> hist;
    int range[2] = {0, 256};
    cv::Mat equMat;
    // myCalcHist_Equalize(image, hist, range, equMat, cv::NORM_MINMAX);
    myCalcHist_Equalize(image, hist, range, equMat, atoi(parameters[2].c_str()));

    // 画直方图，分析直方图
    cv::Mat Hist;
    int channels[1] = {0}; 
    int histSize[1] = {256};
    float range2[2] = {0, 256};
    const float* ranges[1] = {range2}; // 指定每个通道的取值范围

    cv::calcHist(&image, 1, channels, cv::Mat(), Hist, 1, histSize, ranges);
    drawHist(Hist, histSize, cv::Scalar(255,255,255), "Source Histogram");
    
    cv::calcHist(&equMat, 1, channels, cv::Mat(), Hist, 1, histSize, ranges);
    drawHist(Hist, histSize, cv::Scalar(255,255,255), "Equ Histogram");

     // 显示原图
    cv::imshow("Source", image);
    // 显示均衡后的图
    cv::imshow("Equalized", equMat);

    cv::waitKey(0);
    return 0;
}