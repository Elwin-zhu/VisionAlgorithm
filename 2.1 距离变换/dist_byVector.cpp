#include <opencv2/opencv.hpp>
#include <vector>
#include <stdio.h>
#include <unistd.h>
#include <sstream>
#include <algorithm>

using namespace std;

bool InArea(int i, int j, int rows, int cols);
void D4AL(int i,int j, int rows, int cols, vector<vector<int>> &f);
void D4BR(int i, int j, int rows, int cols, vector<vector<int>> &f);


/*
  该距离变换算法中，以图片中的哪些像素为子图是关键，一般src中是经过二值化的全图
*/
void DistanceTransformD4(vector<vector<int>> &src, vector<vector<int>> &f)
{
    int cols = src[0].size();
    int rows = src.size();
    int c =  f[0].size();
    int r = f.size();

    printf("%s:  cols:   %d, rows:   %d \n", __func__, cols, rows);
    printf("%s:  f.cols: %d, f.rows: %d \n", __func__, cols, rows);

    //初始化f，子图的元素值设置成0
    for(int i = 0; i < rows; ++i){
        for(int j = 0; j < cols; ++j){
            if(src[i][j] == 0){ // src数字中为0值的元素为子图
                f[i][j] = 0;
            }
            else{
                f[i][j] = 255;
            }
        }
    }


    //按行遍历图像，从上到下，从左到右
    for (int i = 0; i < rows; ++i){
        for (int j = 0; j < cols; ++j){
            D4AL(i, j, rows, cols, f);
        }
    }

    //按行遍历图像，从下到上，从右到左
    for (int i = rows - 1; i >= 0; --i){
        for (int j = cols - 1; j >= 0; --j){
            D4BR(i, j, rows, cols, f);
        }
    }

    c =  f[0].size();
    r = f.size();
    printf("%s:  f.cols: %d, f.rows: %d \n", __func__, cols, rows);
}

/*
    从左到右，从上到下的变换函数，如下图：
    ▢▢
    ▢▣      ▢q元素，▣p元素，F(p)=min[F(p),D(p,q)+F(q)]
    ▢

*/
void D4AL(int i,int j, int rows, int cols, vector<vector<int>> &f)
{
    //上
    if(InArea(i - 1, j, rows, cols)){
      	f[i][j] = min(f[i][j], 1 + f[i - 1][j]);
    }

    //左上
    if(InArea(i - 1, j - 1, rows, cols)){
	    f[i][j] = min(f[i][j], 2 + f[i - 1][j - 1]);
    }

    //左
    if(InArea(i, j - 1, rows, cols)){
	    f[i][j] = min(f[i][j], 1 + f[i][j - 1]);
    }

    //左下
    if(InArea(i + 1, j - 1, rows, cols)){
	    f[i][j] = min(f[i][j], 2 + f[i + 1][j - 1]);
    }
}

/*
    从左到右，从上到下的变换函数，如下图：
         ▢
       ▣▢      ▢q元素，▣p元素，F(p)=min[F(p),D(p,q)+F(q)]
       ▢▢
*/
void D4BR(int i, int j, int rows, int cols, vector<vector<int>> &f)
{
    //下
    if(InArea(i + 1, j, rows, cols)){
        f[i][j] = min(f[i][j], 1 + f[i + 1][j]);
    }
      
    //右下
    if(InArea(i + 1, j + 1, rows, cols)){
        f[i][j] = min(f[i][j], 2 + f[i + 1][j + 1]);
    }

    //右
    if(InArea(i, j + 1, rows, cols)){
        f[i][j] = min(f[i][j], 1 + f[i][j + 1]);
    }

    //右上
    if(InArea(i - 1, j + 1, rows, cols)){
        f[i][j] = min(f[i][j], 2 + f[i - 1][j + 1]);
    }
}

bool InArea(int i, int j, int rows, int cols)
{
    if (i<0 || i>=rows)
        return false;
    if (j<0 || j>=cols)
        return false;
    return true;
}

std::vector<std::vector<int>> convert_image_to_vector(const cv::Mat& image) 
{
    std::vector<std::vector<int>> image_vector;

    image_vector.reserve(image.rows * image.cols);

    for (int i = 0; i < image.rows; ++i) {
        std::vector<int> row_vector;
        for (int j = 0; j < image.cols; ++j) {
            // 对于彩色图像，可以使用cv::Vec3b来获取三个颜色通道的值
            // 这里我们处理灰度图像，所以用cv::Vec3b也可以，但只取第一个值
            row_vector.push_back(image.at<uchar>(i, j));
        }
        image_vector.push_back(row_vector);
    }
    return image_vector;
}

void myTransform(cv::Mat& imageIn, cv::Mat& imageOut)
{
    std::vector<std::vector<int>> imi, f;
    int cols = imageIn.cols;
    int rows = imageIn.rows;

    //预先分配f
    for(int i=0; i<rows; i++){
        std::vector<int> row_vec;
        for(int j=0; j<cols; j++){
            row_vec.push_back(255);
        }
        f.push_back(row_vec);
    }
    printf("%s: f.cols: %ld, f.rows: %ld\n", __func__, f[0].size(), f.size());
    imi = convert_image_to_vector(imageIn);
    DistanceTransformD4(imi, f);
    //将vector<vector<int>>的数据复制到cv::Mat中
    for(int i = 0; i < f.size(); ++i){
        for(int j = 0; j < f[0].size(); ++j){
            imageOut.at<float>(i, j) = (float)f[i][j];
        }
    }
}

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
    #if 0
        cv::distanceTransform(imageGray, imageThin, CV_DIST_L2, 3);  //距离计算，imageGray矩阵的子集S是字母“ABCD”之外的背景，因为背景是纯黑色，是0，而不是“ABCD”纯白色的字母
    #else
        myTransform(imageGray, imageThin);
    #endif
    
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