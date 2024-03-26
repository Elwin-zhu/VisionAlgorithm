#include <opencv2/opencv.hpp>  //头文件
#include <iostream>
using namespace cv;  //包含cv命名空间
using namespace std;
 
/*
一张很大的图像，某些像素点的值可能有成百上千个，某些像素点可能为0，彼此差距较大。
所以需要进行归一化处理，在特定范围内对像素数据进行缩放。OpenCV4提供了normalize()函数实现多种形式的归一化功能。
OpenCV的函数normalize()的两个作用:调整矩阵的值范围(归一化处理)、规范化矩阵的范数为某个值，究竟函数normalize()发挥什么作用，这取决于参数norm_type取什么值
*/
int main(int argc, char ** argv)
{
	
	vector<double> positiveData = { 2.0, 6.0, 7.0, 8.0, 10.0 };
	vector<double> normalizedData_l1, normalizedData_l2, normalizedData_inf, normalizedData_minmax;
	// NORM_L1 模式下，累加positiveData中的所有数据的绝对值之和为norm=|2.0|+|6.0|+|7.0|+|8.0|+|10.0|=33.0
	// normalizedData_l1[i] = 5.0*(positiveData[i]/norm)
	normalize(positiveData, normalizedData_l1, 5.0, 100.0, NORM_L1);
	cout <<"normalizedData_l1: " << endl;
	for (size_t i = 0; i < normalizedData_l1.size(); i++)
	{
		cout << normalizedData_l1[i] << ",\t";
	}
 
	// NORM_L2 模式下，累加positiveData中的所有数据的平方和，然后开方，norm=2.0**2+6.0**2+7.0**2+8.0**2+10.0**2=15.906
	// normalizedData_l2[i] = 5.0*(positiveData[i]/norm)
	normalize(positiveData, normalizedData_l2, 5.0, 100.0, NORM_L2);
	cout << "\nnormalizedData_l2: " << endl;
	for (size_t i = 0; i < normalizedData_l2.size(); i++)
	{
		cout << normalizedData_l2[i] << ",\t";
	}
 
    //NORM_INF 模式下，取出positiveData中最大的元素而得到norm，可以用函数minMaxLoc求出
	//normalizedData_inf[i] = 5.0*(positiveData[i]/norm)
	normalize(positiveData, normalizedData_inf, 5.0, 100.0, NORM_INF);
	cout << "\nnormalizedData_inf: " << endl;
	for (size_t i = 0; i < normalizedData_inf.size(); i++)
	{
		cout << normalizedData_inf[i] << ",\t";
	}
 
	//Norm to range [5.0;100.0]
    //NORM_MINMAX 模式下，代入的beta起效，100.0为b，5.0为a。
    //根据等比例缩放公式：(dst(i) - a)/(b - a) = (src(i) - min(src))/(max(src) - min(src))，得到：
	//normalizedData_minmax[i] = (b-a)*(positiveData[i]-min(positiveData))/(max(positiveData)-min(positiveData)) + a
	normalize(positiveData, normalizedData_minmax, 5.0, 100.0, NORM_MINMAX);
	cout << "\nnormalizedData_minmax: " << endl;
	for (size_t i = 0; i < normalizedData_inf.size(); i++)
	{
		cout << normalizedData_minmax[i] << ",\t";
	}
	cout <<  endl;
 
	//等待任意按键按下
	waitKey(0);
	return 0;
}