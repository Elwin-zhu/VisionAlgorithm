
#include <opencv2/opencv.hpp>
#include <iostream>

/*
  图像: 亮度、对比度、饱和度、锐化
  图像亮度、对比度、饱和度和锐化之间并不是彼此独立的，改变其中一个特征可能会同时引起图像其他特征的变化，至于变化的程度取决于图像本身的特性.
  1、亮度
  图像亮度通俗理解便是图像的明暗程度，数字图像f(x，y) ，如果灰度值在[0，255]之间，则 f 值越接近0亮度越低，f 值越接近255亮度越高。而且我们也要把亮度和对比度区分开来，正如上述提的对比度指的是最高和最低灰度级之间的灰度差。
  过度增加亮度导致阴影赶上了高光，因为最大灰度值是固定的，所以最低灰度值快赶上了最大灰度值，因此影响了图片的饱和度、对比度和清晰度.
  2、饱和度
  饱和度指的是图像颜色种类的多少，上面提到图像的灰度级是[Lmin，Lmax]，则在Lmin、Lmax 的中间值越多，便代表图像的颜色种类多，饱和度也就更高，外观上看起来图像会更鲜艳，调整饱和度可以修正过度曝光或者未充分曝光的图片。使图像看上去更加自然
  3、锐化
  图像锐化是补偿图像的轮廓，增强图像的边缘及灰度跳变的部分，使图像变得清晰。图像锐化在实际图像处理中经常用到，因为在做图像平滑，图像滤波处理的时候经过会把丢失图像的边缘信息，通过图像锐化便能够增强突出图像的边缘、轮廓.
  HSV/HSB颜色空间：H是色调，S是饱和度，I是强度，有许多种HSX颜色空间，其中的X可能是V,也可能是I
  HSB（Hue, Saturation, Brightness）颜色模型，B是亮度，这个颜色模型和HSL颜色模型同样都是用户台式机图形程序的颜色表示，用六角形锥体表示自己的颜色模型。
  HSL（Hue, Saturation, Lightness）颜色模型, H是色调，S是饱和度，L是明度
*/

int main()
{
    // 读取图像
    cv::Mat image = cv::imread("test1.jpg", cv::IMREAD_GRAYSCALE);
    if (image.empty()) {
        std::cout << "图像加载失败" << std::endl;
        return -1;
    }
    printf("%d, %d, %d, %d\n", image.rows, image.cols, image.dims, (int)image.step);
    fflush(stdout);
    // cv::imwrite("./test2.bmp", image);

    cv::Mat equ;
    ////根据单通道1 uchar级数来均衡化图片，效果明显，//单通道灰度图片最小级数为0，最大为255
    //注：直方图均衡化只能在单通道图片进行，直方图均衡化的主要作用是提高图片的饱和度，即让直方图尽量扩散分布在0-255的亮度区间
    cv::equalizeHist(image, equ);
 
 
    // 显示原图
    cv::imshow("Source", image);
    //显示均衡后的图
    cv::imshow("Equalized", equ);
    cv::waitKey(0);
    return 0;
}