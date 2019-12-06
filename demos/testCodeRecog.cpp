// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

// This code is also subject to the license terms in the LICENSE_WillowGarage.md file found in this module's directory

#define CV__ENABLE_C_API_CTORS // enable C API ctors (must be removed)


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

#include "DllmvInterface.h"
#include "mvSDK_interface.h"
#include "cvInterface.h"



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
	const char *path1 = "D:/pic/qr_code2.jpg";
	const char *path2 = "./pic/qr_code2.jpg";

	Mat img;
	VideoCapture cap;
	int index = 0;

	//img = imread(path1);
	//if (img.empty())
	//{
	//	img = imread(path2);
	//}
 

	cap.open(index);
	if (!cap.isOpened())
			cap.open(0);
	int n;

	if (!cap.isOpened())
	{
		cerr << "Unable to open video capture." << endl;
		return -1;
	}

	for (;;)
	{
		cap >> img;

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
		waitKey(20);
	}

	return(0);
}