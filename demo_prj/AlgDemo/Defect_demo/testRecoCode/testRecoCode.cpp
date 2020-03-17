/*****************************************************

tensorflow c++ 接口部署深度学习模型

Author:
	WSN@2019
******************************************************/

#include <opencv2/opencv.hpp>
#include <opencv2/core/types_c.h>
#include <opencv2/core/core_c.h>
#include "opencv2/highgui/highgui_c.h"
#include "opencv2/imgproc/imgproc_c.h"
#include "opencv2/video/video.hpp"
#include <vector>
#include <fstream>

#include <opencv2/opencv.hpp>
#include <opencv2/core/types_c.h>
#include <opencv2/core/core_c.h>
#include "opencv2/highgui/highgui_c.h"
#include "opencv2/imgproc/imgproc_c.h"
#include "opencv2/video/video.hpp"
#include <vector>
#include <fstream>

#include "mvSDK_interface.h"

#pragma comment(lib,"LibX-Vision_x64.lib")

using namespace std;
using namespace cv;

typedef struct
{
	int valid;                 /*  symbol valid flag */
	std::string type;          /*  decode type       */
	int quality;               /*  symbol quality    */
	std::string content;       /*  decode content    */
	mvRect rect;               /*  symbol rectangle  */
	mvPoint center;            /*  symbol center     */
	vector<mvPoint>pnts;       /*  symbol location points   */
}symObj;

/*
qr-code,bar-code recognition.
@img     input, image to detect
@objs    output, object

reutrn
*/
extern void mvCodeRecognition(Mat& img, vector<symObj>& objs);

int main(int argc, char **argv)
{
	//const char *path1 = "../../images/qr_code2.jpg";

	Mat img;
	VideoCapture cap;
	int index = 0;
	int flag;
	//img = imread(path1);

	string dir_path = "../../images/code";

	vector<string> file_list;
	cv::glob(dir_path, file_list);    	/* 从文件夹路径读取所有测试图片 */

	for (int ii = 0; ii < file_list.size(); ii++)

	if (file_list.size())
		flag = 0;
	else
	{
		cap.open(index);
		if (!cap.isOpened())
			cap.open(0);
		int n;

		if (!cap.isOpened())
		{
			cerr << "Unable to open video capture." << endl;
			return -1;
		}
		flag = 1;
	}

	index = 0;
	for (;;)
	{
		if (flag)
			cap >> img;
		else
		{
			if (index < file_list.size())
			{
				String pp = file_list[index++];
				img = imread(pp);
			}
			else
				index = 0;
		}

		vector <symObj> objs;

		mvCodeRecognition(img, objs);

		for (int ii = 0; ii < objs.size(); ii++)
		{
			Point c1, c2;
			c1.x = objs[ii].rect.up_left.x;
			c1.y = objs[ii].rect.up_left.y;
			c2.x = objs[ii].rect.dw_right.x;
			c2.y = objs[ii].rect.dw_right.y;
			rectangle(img, Rect(c1, c2), Scalar(0, 0, 255), 2);
			c2.x = objs[ii].center.x;
			c2.y = objs[ii].center.y;
			circle(img, c2, 5, Scalar(0, 255, 0), 2);
			cout << "type: " << objs[ii].type << endl;
			cout << "content:" << objs[ii].content << endl;
			int font_face = cv::FONT_HERSHEY_COMPLEX;
			double font_scale = 1;
			cv::putText(img, objs[ii].content, c1, font_face, font_scale,
				Scalar(0, 255, 0), 1, 8, 0);
		}

		imshow("decode", img);
		if (waitKey(1000) == 27)
			break;
	}

	waitKey();
	cout << "exit." << endl;

	return(0);
}