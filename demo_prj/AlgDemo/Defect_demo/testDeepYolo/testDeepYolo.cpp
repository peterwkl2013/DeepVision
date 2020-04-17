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
	String class_id;    /* class id */
	int index;          /* template index */
	float score;        /* score */
	float area;         /* area of this object */
	float angle;        /* angle of this object */
	Point2f r_center;   /* rotation center */
	Rect  rect;         /* rectangle location */
	int valid;          /* roi valid */
}yobjectResult;

class mvYoloDetection
{
public:

	mvYoloDetection(String model, int width = 512, int height = 512,
		float _score = 0.6, int _draw = 0, float ovthres = 0.45)
	{
		models_path = model;
		score_thres = _score;
		draw_flag = _draw;
		in_width = width;
		in_height = height;
		draw_flag = _draw;
		overlap_thres = ovthres;
		myinst = NULL;
	}

	~mvYoloDetection();

	int process(Mat& img, vector<yobjectResult>& mres, float score_thres = 0.6);
	/*
	if the detection object's location is out of the roi map, will be ignored
	@input ppts     polygon points
	@input, width   the detection roi image area
	@input, height  the detection roi image area
	*/
	void setDetRoi(vector<vector<Point>>& ppnts, int width, int height, int is_show);
public:
	String models_path;              /* model file to read from */
	int in_width;                    /* image input width: default,512 */
	int in_height;                   /* image_input heght: default,512 */
	Mat det_img;                     /* detect image (input) */
	Mat roi_map;                     /* detection roi map (gray image) */
	int draw_flag;                   /* internal draw flag */
	float overlap_thres;             /* multi-object with the same class id remove threshold,  rect overap */
	float score_thres;               /* score threshold */
private:
	int test;
	class mvYolo;
	mvYolo* myinst;
};

static void drawPred(Mat& frame, yobjectResult& obj)
{
	int top;
	int classId;
	float conf;
	Rect rc;

	classId = obj.index;
	conf = obj.score;
	rc = obj.rect;

	//Draw a rectangle displaying the bounding box
	if (obj.valid)
		rectangle(frame, rc, Scalar(0, 0, 255));
	else
		rectangle(frame, rc, Scalar(127, 127, 127));

	//Get the label for the class name and its confidence
	string label = format("%d", classId);
	//string label = format("%d", classId);
	//Display the label at the top of the bounding box
	int baseLine;
	Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
	top = max(rc.tl().y, labelSize.height);

	putText(frame, label, Point(rc.tl().x, top - 2), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0));
}


void setDetRoi(vector<vector<Point>>& ppts, int width, int height, int is_show)
{
	//CV_EXPORTS_W void polylines(InputOutputArray img, InputArrayOfArrays pts,
	//	bool isClosed, const Scalar& color,
	//	int thickness = 1, int lineType = LINE_8, int shift = 0);
	Mat img(height, width, CV_8UC1, Scalar(0));

	vector <Point> tmp;

	for (int i = 0; i < ppts.size(); i++)
	{
		tmp = ppts[i];


		//const Point *ps = &ppts[0];
		////ps[0][0] = ppts[0];
		////ps[0][1] = ppts[1];
		////ps[0][2] = ppts[2];
		////ps[0][3] = ppts[3];
		////ps[0][4] = ppts[4];
		//const Point *ppt[1] = { &ps[0] };
		//int npt[] = { ppts.size() };

		const int hull_count = (int)tmp.size();
		const cv::Point* hull_pts = &tmp[0];


		cv::fillPoly(img, &hull_pts, &hull_count, 1, Scalar(i+1));


		cv::polylines(img, tmp, 1, Scalar(255), 1, 8, 0);
		if (is_show)
			cv::imshow("poly", img);

		if (is_show)
		{
			Mat cc;
			cv::threshold(img, cc, 0, 255, THRESH_BINARY);
			cv::imshow("poly-roi", cc);
		}

		waitKey();
	}

	return;
}
int main()
{
	//mvYoloDetection test_yolo("../../models/test_yolov3_wenli.pb");
	//string dir_path = "../../images/wenli";

	mvYoloDetection test_yolo("../../models/mvPlate_aug.pb");
	string dir_path = "../../images/plate";

	vector<string> file_list;
	cv::glob(dir_path, file_list);    	/* 从文件夹路径读取所有测试图片 */

	for (int ii = 0; ii < file_list.size(); ii++)
	{
		String pp = file_list[ii];

		Mat img = imread(pp, IMREAD_COLOR);
		Mat input0, input;

		cout << "img:" << pp << endl;

		imshow("input", img);
		vector<Point> ppnts;

		vector<vector<Point>> pls;

		Point p1;
		int c1 = img.cols / 2;
		int c2 = img.rows / 2;

		p1 = Point(c1-100, c2 - 100);
		ppnts.push_back(p1);
		p1 = Point(c1 + 20, c2 - 100);
		ppnts.push_back(p1);
		p1 = Point(c1 + 20, c2 + 100);
		ppnts.push_back(p1);
		p1 = Point(c1 - 100, c2 + 100);
		ppnts.push_back(p1);

		pls.push_back(ppnts);

		ppnts.clear();
		c1 = 20, c2 = 80;

		p1 = Point(c1 - 20, c2 - 20);
		ppnts.push_back(p1);
		p1 = Point(c1 + 20, c2 - 20);
		ppnts.push_back(p1);
		p1 = Point(c1 + 20, c2 + 20);
		ppnts.push_back(p1);
		p1 = Point(c1 - 20, c2 + 20);
		ppnts.push_back(p1);
		pls.push_back(ppnts);
		//setDetRoi(pls, img.cols, img.rows, 1);

		//p1 = Point(320, 180);
		//ppnts.push_back(p1);
		//p1 = Point(235, 465);
		//ppnts.push_back(p1);

		vector<yobjectResult> mres;
		double ptime;
		int64 nTick, nTick2;
		nTick = getTickCount();
		
		test_yolo.setDetRoi(pls, img.cols, img.rows, 1);
		test_yolo.process(img, mres);

		nTick2 = getTickCount();
		ptime = ((double)nTick2 - nTick)*1000. / getTickFrequency();
		printf("process time: %fms\n", ptime);

		/* 绘制结果 */
		for (int jj = 0; jj < mres.size(); jj++)
		{
			yobjectResult obj = mres[jj];
			vector <Point> pnts;
			Point2f obj_center;

			/* draw detection */
			drawPred(img, obj);
			drawPred(test_yolo.roi_map, obj);
		}

		imshow("result", img);
		imshow("roi_map", test_yolo.roi_map);
		waitKey();
	}


	return 1;
}
