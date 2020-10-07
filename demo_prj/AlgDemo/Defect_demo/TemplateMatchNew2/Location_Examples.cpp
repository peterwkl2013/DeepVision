/****************************************************

	 template detection algorithm examples
	              
@BY X-VISION
@2020/09/26
*****************************************************/

#include <vector>
#include <fstream>
#include <opencv2/opencv.hpp>
#include "opencv2/imgproc/imgproc_c.h"

using namespace cv;
using namespace std;

static RNG rng;
 
/* match result */
typedef struct transmat
{
	int type;             /* 0: 2x3, 1: 3x3, 2: 4x4 */
	double pdata[16];     /* transform matrix 2x3, 3x3, 4x4 */
}mvTransfor;

typedef struct
{
	Point2f pnts[4];         /* 4-point represent of this box */
	Point2f center;          /* center point of this box */
	float   angle;           /* angle of this box */
	float   width;           /* width */
	float   height;          /* height */
	float   area;            /* area of this box */
}mvFBox;

typedef struct
{
	String class_id;         /* class name */
	int index;               /* template index */
	int obj_index;           /* object index */
	float score;             /* score */
	float angle;             /* angle of this matched object*/
	float scale;             /* scale factor */
	Point2f r_center;        /* rotation center */
	Rect  rect;              /* rectangle location */
	mvFBox bbox;             /* box olcation */
	int x_offset, y_offset;
	int roi_index;            /* roi index */
	int valid;                /* roi valid */
	mvTransfor tmat;          /* transform mat */
	int track_enable;         /* track enable */
	int is_track;             /* is beging tracking */
	int track_id;             /* is beging tracking */
	int obj_data;             /* object data */
	vector<Point>track_pnts;  /* traking center point */
}matchResult;

class  mv2DTemplateMatch
{
public:
	mv2DTemplateMatch();
	~mv2DTemplateMatch();

	/*
	@input, temps, in the tempalte image
	@input, class_ids class name, if empty, default id will be matched
	@input, polygon point in the image
	*/
	int trainTemplate(Mat& templ, String& class_id, Rect& roi, Mat& mask = Mat());
	int trainTemplate(Mat& templ, String& class_id, vector<vector<Point2f>>& loc_regions, Mat& mask = Mat());
	/*
	@input, template image
	@input, class_ids class name, if empty, default id will be matched
	@input, roi int the template image
	*/
	int trainTemplate(vector<Mat>& templs, vector<String>& class_ids, vector<Rect>& roi_rect, vector<Mat>& mask = vector<Mat>());
	int trainTemplateFast(vector<Mat>& templs, vector<String>& class_ids, vector<Rect>& roi_rect, vector<Mat>& mask = vector<Mat>());

	/*
	@input, det_img  detection image,color-image
	@input, class_ids class name, if empty, default id will be matched
	@input, conf  score confidence (0~100)
	@output, mres, match result
	@reutrn true or false or -1 invalid
	*/
	int matchTemplate(Mat& det_img, vector<cv::String>& class_ids, vector<matchResult>& mres, float score = 0.80);

	/* two metho to refine location */
	void refineObjectPos(Mat& temp, Mat& det, matchResult& mres);
	void refineObjectPos(Mat& det, vector<matchResult>& v_mres);

	/*input, ranges,  +- ranges  (0~360.0 degree)
	  eg.  +-45 ( 0~45, 360 ~ 315)
	*/
	void setLimitAngleRanges(float ranges = 45.0) { angle_ranges = ranges; }

	/*
	detection roi map(the same size of the 'det_img' image)
	@input, width
	@input, height
	*/
	void setDetRoiMap(vector<vector<Point>>& plys, int width, int height, int is_show = 0);
	void setDetRoiMap(Mat& map) { roi_map = map; }
	Mat  getDetRoiMap() { return roi_map; }
	Mat  createDetRoiMap(vector<vector<Point>>& plys, vector<int>& type, int width, int height, int is_show = 0);

	/*
	@output, the hmat, get the template affine transform matrix: 2x3.
	*/
	void getAFTransform(matchResult& mres, double* hmat);
	/*
	@tempp, input,  template points in the template image.
	@dstpp, output, output points in the detection image.
	*/
	void calcPointPosition(vector<Point>& tempp, vector<Point>& dstpp, double* phmat);
	void calcPointPosition(vector<Point>& tempp, vector<Point2f>& dstpp, double* phmat);
	void calcPointPosition(vector<Point2f>& tempp, vector<Point2f>& dstpp, double* phmat);

	/*
	when change a new class id, need to reset
	*/
	void reset();

	/* object tracking functions */
	void enableObjectTracking(int bhv) { tracking = bhv; }
	void initObjectTracker(int numtrack = 10, int start_frame = 5, int tr_len = 20, int track_del = 10, float thres = 0.6);
	void setTrakingObjectData(vector<matchResult>& mres);

public:
	String config_path;              /* template feature to store or to read from */
	Mat det_img;                     /* detect image (input) */
	Mat temp_img;                    /* template image (input) */
	Mat roi_map;                     /* detection roi map (gray image) */
	int sel_mode;                    /* select feature mode: 0-corners, 1-select edge, 2-select edge,no limited featres */
	int num_detect;                  /* num object to detect in one image */
	vector<float> scale_ranges;      /* scale factor(0.8~1.5) for training,  now only support scale = 1.0 */
	float angle_ranges;              /* min-max angles rangles, according to 0.0 */
	float angle_step;                /* angle step when train template */
	int   num_features;              /* feature to extract in the template */
	float min_thres;                 /* to control the feature response */
	float max_thres;                 /* to control the feature response */
	int   pos_refine;                /* set true if refine object position, also can use 'refineObjectPos' manually */
	int   tracking;                  /* set true if tracking the detect-object */
	int   num_track;                 /* num object to tracking */
	int   tracking_start_frame;      /* start tracking after num frames */
	int   track_length;              /* tracking length */
	int   track_del;                 /* tracking lenght to del */
	float track_thres;               /* tracking match score */

	/* when the template size(width, height) is too large  , it is necessary
	to set the filter size: 3, 5, 7, 9,11, to filter the features, else set to 0 , when have enough features */
	int   filter_size;
	float overlap_thres;            /* multi-object with the same class id remove threshold,  rect overap */
private:
	int test;
	class mympl;
	mympl* mpl;
};

#pragma comment(lib,"LibDeepVision_x64.lib")

const int ARROW_LENGHT = 120;
const int TRIANGLE_HEIGHT = 25;
const double TRIANGLE_DEGREES = 0.2;

void DrawArrow(cv::Mat& show_img, Point2f p1, Point2f p2, cv::Scalar color)
{
	float direction_angle, x1, y1, x2, y2;

	direction_angle = std::atan2(p2.y - p1.y, p2.x - p1.x) + 3.1415926;

	x1 = p2.x + TRIANGLE_HEIGHT * cos(direction_angle - TRIANGLE_DEGREES);
	y1 = p2.y + TRIANGLE_HEIGHT * sin(direction_angle - TRIANGLE_DEGREES);
	x2 = p2.x + TRIANGLE_HEIGHT * cos(direction_angle + TRIANGLE_DEGREES);
	y2 = p2.y + TRIANGLE_HEIGHT * sin(direction_angle + TRIANGLE_DEGREES);

	cv::line(show_img, p2,cv::Point(x1, y1), color, 2);
	cv::line(show_img, p2,cv::Point(x2, y2), color, 2);
}


void DrawCoordTemplate(cv::Mat& show_img, const matchResult& obj, float scale)
{
	float angle = obj.angle;
 
	int wh = ARROW_LENGHT;

	Point2f p1, p2;

	p2 = obj.bbox.center;
	p2.x += ARROW_LENGHT;
	cv::line(show_img, obj.bbox.center, p2, cv::Scalar(0, 0, 255), 3);
	DrawArrow(show_img, obj.bbox.center, p2, cv::Scalar(0, 0, 255));


	p2 = obj.bbox.center;
	p2.y -= ARROW_LENGHT;

	cv::line(show_img, obj.bbox.center, p2, cv::Scalar(0, 255, 0), 3);
	DrawArrow(show_img, obj.bbox.center, p2, cv::Scalar(0, 255, 0));

	cv::circle(show_img, obj.bbox.center, 3, cv::Scalar(0, 255, 255), -1);
	char str[128]{ 0 };

	sprintf_s(str, "(%d,%d)", obj.bbox.center.x, obj.r_center.y);
	cv::putText(show_img, str, obj.bbox.center, CV_FONT_HERSHEY_COMPLEX_SMALL, 0.4 / scale, cv::Scalar(255, 255, 0), 0.5 / scale);
}


typedef struct
{
	float x;                /* x position */
	float y;                /* y position */
}mvFPoint;

/***
get the map point from other image[homography].

@inp,    input, the detect image point position
@hmat,   input, 3x3 perspective mat data[9]
@type,   input, type = 0,  calc the template image map point
type = 1,  calc the template image map point
@outp,   ouput, the detect image point position

*/
void mvMap2DPoint(mvFPoint inp, double *hmat, int type, mvFPoint &outp);

int main(int argc, char * argv[])
{

	int ret;
	mv2DTemplateMatch  my_templ;
	vector<matchResult>mres;

	/* 算法处理尺度 scl */
	float scl = 1.0;
	float conf = 0.8f;
	int trainFlag = 1;

	/* 算法参数设置 */
	my_templ.num_detect = 1;
	my_templ.filter_size = 3;
	my_templ.num_features = 64;
	my_templ.angle_step = 1;
	my_templ.sel_mode = 1;

	String ssss = "11,22,33,44,55";
	vector <String>rssss;

	char *tokenptr = strtok((char*)ssss.c_str(), ",");
	while (tokenptr != NULL)
	{
		std::string tk = tokenptr;
		rssss.emplace_back(tk);
		tokenptr = strtok(NULL, ",");
	}

	const char *path1 = "D:\\bd2_750x1024b.bmp";
	const char *path2 = "D:\\bd2_750x1024b.bmp";

	Mat tmpl = imread(path1);
	Mat det_img1 = imread(path2);

	//test_roi(tmpl);

	resize(det_img1, tmpl, Size(det_img1.cols*scl, det_img1.rows*scl));
	det_img1 = tmpl.clone();

	vector <Mat> img_dets, img_tmpls;
	vector<String> class_ids;

	String id1 = "0";

	Point temp_offset1(400 * scl, 400 * scl);
	Point temp_offset2(200 * scl, 400 * scl);


	//Point temp_offset1(860 * scl, 860 * scl);
	//Point temp_offset2(200 * scl, 400 * scl);

	Point temp_offset;
	double matrix4x4[16];
	double matrix3x3[9];

	vector <Point2f> offset;


	/* 建立模板输入图像以及类别: eg: "id1, id2,id3..." */
	Rect tmp_roi = Rect(temp_offset1.x, temp_offset1.y, 200 * scl, 200 * scl);
	vector<Rect> tmp_rois;

	offset.push_back(temp_offset1);
	class_ids.push_back("id1");
	img_dets.push_back(det_img1);
	tmp_rois.push_back(tmp_roi);

	if (trainFlag)
	{/* 模板训练 */
		vector<Mat> mask_rois;
		cout << "train template..." << endl;
		TickMeter tick;
		tick.start();
		ret = my_templ.trainTemplate(img_dets, class_ids, tmp_rois, mask_rois);
		//ret = my_templ.trainTemplateFast(img_dets, class_ids, tmp_rois);
		if (ret <= 0)
			cout << "train template error! please change \
			the image size，filter_size or num_features size" << endl;

		tick.stop();
		//1. 33288ms
		printf("trainning time:%.3f\n", tick.getTimeMilli());
	}

	rng = RNG((unsigned)time(NULL));

	/* 模板检测任务开始 */
	mvFPoint mm1, mm2;

	vector <float> cc1, cc2, cc3;

	float min1, max1, min3, max3;
	min1 = max1 = min3 = max3 = 0;
	int count = 0;
	for (size_t ii = 0; ii < img_dets.size(); ii++)
	{
		double hmat[9];

		Mat det_img = img_dets[ii];
		Point2f center(det_img.cols / 2, det_img.rows / 2);
		Mat img_roate;
		Mat det, det_affine;
		Mat tmp_copy, tmp_gray;
		Mat dst_copy, dst_gray;

		for (float kk = 0; kk < 359; kk += 1.8)
		{
			int a = rng.uniform(20, 100);
			int b = rng.uniform(20, 100);
			Point2f oo = Point(a, a);
			{

				mm1.x = (tmp_roi.tl().x + tmp_roi.br().x) / 2.0;
				mm1.y = (tmp_roi.tl().y + tmp_roi.br().y) / 2.0;

				img_roate = cv::getRotationMatrix2D(center, kk, 1);//center:模板整图中心
				det_img = img_dets[ii].clone();

				//检测中坐标om1
				mvMap2DPoint(mm1, (double*)img_roate.data, 0, mm2);//mm2:模板区域中心绕模板整图中心旋转kk度后的坐标,即误差参考点

				//circle(det_img, Point(mm2.x, mm2.y), 3, Scalar(0, 255, 0), -1);

				det = det_img;
				cv::warpAffine(det_img, det, img_roate, det_img.size());
				circle(det, Point(mm2.x, mm2.y), 3, Scalar(0, 255, 0), -1);

			}

			double ptime;
			int64 nTick, nTick2;
			nTick = getTickCount();
			Point p1;
			vector <Point> ppnts;
			vector<vector <Point>> pls;

			int c1 = det.cols / 2;
			int c2 = det.rows / 2;

			p1 = Point(c1 - 60, c2 - 120);
			ppnts.push_back(p1);
			p1 = Point(c1 + 120, c2 - 120);
			ppnts.push_back(p1);
			p1 = Point(c1 + 120, c2 + 120);
			ppnts.push_back(p1);
			p1 = Point(c1 - 120, c2 + 120);
			ppnts.push_back(p1);
			pls.push_back(ppnts);

			c1 += 80, c2 += 160;
			ppnts.clear();
			p1 = Point(c1 - 80, c2 - 80);
			ppnts.push_back(p1);
			p1 = Point(c1 + 80, c2 - 80);
			ppnts.push_back(p1);
			p1 = Point(c1 + 80, c2 + 80);
			ppnts.push_back(p1);
			p1 = Point(c1 - 80, c2 + 80);
			ppnts.push_back(p1);

			pls.push_back(ppnts);

			static int  counter = 0;
			counter++;

			if (counter < 0)
			{
				counter = counter;
				continue;
			}

			//if (counter > 15 && counter < 50)
			//{ 
			//	det = cv::Mat::zeros(det.rows, det.cols, det.type());
			//	//continue;
			//}
			//else
			//{
			//	if (counter > 100)
			//		counter = 0;
			//}
			//imshow("edge0", det);
			//edgePreservingFilter(det, det, 1, 60.f, 0.4f);
			//imshow("edge", det);
			//waitKey();

			/* 设置ROI检测区域，可以同时设置多个roi_index为ROI区域编号(1,2,3,4...)*/
			//my_templ.setDetRoiMap(pls, det.cols, det.rows);

			/* conf为相似度分数设置: 0~1.0 */
			static int ffg = 0;
			double avg_rough_time = 0.f;
			if (ffg == 0)
			{
				TickMeter tick;
				tick.start();
				ret = my_templ.matchTemplate(det, class_ids, mres, conf);
				tick.stop();

				avg_rough_time = tick.getTimeMilli();

			}

			//nTick2 = getTickCount();
			//ptime = ((double)nTick2 - nTick)*1000. / getTickFrequency();

			tmp_copy = img_dets[0].clone();
			dst_copy = det.clone();
			det_affine = det.clone();

			if (tmp_copy.channels() > 1)
				cv::cvtColor(tmp_copy, tmp_gray, COLOR_BGR2GRAY);
			else
				tmp_gray = tmp_copy;
			if (tmp_copy.channels() > 1)
				cv::cvtColor(dst_copy, dst_gray, COLOR_BGR2GRAY);
			else
				dst_gray = dst_copy;

			vector<matchResult>mres2 = mres;

			Point2f pre_center;
			float prev_angle;

			if (mres.size() > 0)
			{
				pre_center = mres[0].bbox.center;
				prev_angle = mres[0].bbox.angle;
			}
			TickMeter tick_refine;
			tick_refine.start();
			my_templ.refineObjectPos(dst_gray, mres);
			tick_refine.stop();
			double tt = tick_refine.getTimeMilli();
			printf("alg match time: %f ms, icp refine time:%f ms, total = %f ms\n", avg_rough_time, tt, tt + avg_rough_time);
			/* 显示结果匹配结果 */
			int max_ffg = 0;
			for (int jj = 0; jj < mres.size(); jj++)
			{
				matchResult obj = mres[jj];
				//my_templ.refineObjectPos(dst_gray, obj);
				vector <Point> pnts;

				pnts.reserve(4);
				pnts.push_back(Point(obj.bbox.pnts[0].x, obj.bbox.pnts[0].y));
				pnts.push_back(Point(obj.bbox.pnts[1].x, obj.bbox.pnts[1].y));
				pnts.push_back(Point(obj.bbox.pnts[2].x, obj.bbox.pnts[2].y));
				pnts.push_back(Point(obj.bbox.pnts[3].x, obj.bbox.pnts[3].y));

				vector<Point2f>  feature_points;

				Mat affine_mat = Mat(Size(3, 2), CV_64FC1, (void*)obj.tmat.pdata);

				/* draw location */
				for (int i = 0; i < 4; i++)
				{
					int next = (i + 1 == 4) ? 0 : (i + 1);
					cv::line(dst_copy, Point(obj.bbox.pnts[i].x, obj.bbox.pnts[i].y),
						Point(obj.bbox.pnts[next].x, obj.bbox.pnts[next].y), Scalar(0, 0, 255), 2);
					cv::String txt;
					char str[512];

					Point temp_offset;
					if (obj.class_id == "id1")
						cv::putText(dst_copy, str, Point(10, 40), CV_FONT_HERSHEY_COMPLEX, 0.5, Scalar(0, 255, 0));
					else
						cv::putText(dst_copy, str, Point(10, 80), CV_FONT_HERSHEY_COMPLEX, 0.5, Scalar(0, 255, 0));
				}

				for (int i = 0; i < 4; i++)
				{
					int next = (i + 1 == 4) ? 0 : (i + 1);
					//if (obj.valid)
					//{
					//	cv::line(dst_copy, Point(obj.bbox.pnts[i].x, obj.bbox.pnts[i].y),
					//		Point(obj.bbox.pnts[next].x, obj.bbox.pnts[next].y), Scalar(255, 0, 0), 2);
					//	cv::line(my_templ.roi_map, Point(obj.bbox.pnts[i].x, obj.bbox.pnts[i].y),
					//		Point(obj.bbox.pnts[next].x, obj.bbox.pnts[next].y), Scalar(60, 60, 60), 2);
					//	circle(my_templ.roi_map, Point(obj.bbox.center.x, obj.bbox.center.y), 3, Scalar(60 * obj.roi_index, 60 * obj.roi_index, 60 * obj.roi_index), -1);
					//}
					//else
					{
						cv::line(dst_copy, Point(obj.bbox.pnts[i].x, obj.bbox.pnts[i].y),
							Point(obj.bbox.pnts[next].x, obj.bbox.pnts[next].y), Scalar(0, 255, 0), 2);
						circle(my_templ.roi_map, Point(obj.bbox.center.x, obj.bbox.center.y), 3, Scalar(255, 255, 255), -1);
					}
					cv::String txt;
					char str[512];

					sprintf_s(str, "index: %d, score = %.2f, angle = %.1f(%.1f)", counter, obj.score, prev_angle,
						obj.angle);
					Point temp_offset;
					if (obj.class_id == "id1")
						cv::putText(dst_copy, str, Point(10, 40), CV_FONT_HERSHEY_COMPLEX, 0.5, Scalar(0, 255, 0));
					else
						cv::putText(dst_copy, str, Point(10, 80), CV_FONT_HERSHEY_COMPLEX, 0.5, Scalar(0, 255, 0));

				}

				float dif, anglediff, dif0, anglediff0;
				float a, b, a0, b0;

				a0 = (mm2.x - pre_center.x) * (mm2.x - pre_center.x);
				b0 = (mm2.y - pre_center.y) * (mm2.y - pre_center.y);
				dif0 = sqrt((float)(a0 + b0));
				anglediff0 = fabs(kk - prev_angle);


				a = (mm2.x - obj.bbox.center.x) * (mm2.x - obj.bbox.center.x);
				b = (mm2.y - obj.bbox.center.y) * (mm2.y - obj.bbox.center.y);

				dif = sqrt((float)(a + b));
				anglediff = fabs(kk - obj.angle);


				cc1.push_back(dif);
				cc2.push_back(dif / sqrt(2.0));
				cc3.push_back(anglediff);


				float sum1, sum2, sum3;
				sum1 = sum2 = sum3 = 0;
				for (int i = 0; i < cc1.size(); i++)
				{
					sum1 += cc1[i];

					sum2 += cc2[i];
					sum3 += cc3[i];

				}

				sum1 /= cc1.size();
				sum2 /= cc2.size();
				sum3 /= cc3.size();

				if (min1 > dif)
				{
					min1 = dif;
				}
				if (max1 < dif)
				{
					max1 = dif;
				}

				if (min3 > anglediff)
				{
					min3 = anglediff;
				}
				if (max3 < anglediff)
				{
					max3 = anglediff;
				}
				//printf("处理时间: %f 毫秒, 偏离真实目标 = %.4f(平均:%.4f)像素, pix = %.4f(%.4f)  angle_dif = %.4f\n", ptime, dif, sum1, dif/sqrt(2.0), sum2, fabs(kk-obj.angle));

				printf("像素精度 = %.4f(%.4f)(平均:%.4f,最小:%.4f,最大:%.4f),角度精度 = %.4f(平均:%.4f,最小:%.4f,最大:%.4f),粗匹配耗时:%f, 精匹配耗时:%f\n", dif, dif0, sum1, min1, max1, anglediff, sum3, min3, max3, avg_rough_time, tick_refine.getTimeMilli());
				count++;
				DrawCoordTemplate(dst_copy, obj, scl);
				circle(dst_copy, Point(mm2.x, mm2.y), 3, Scalar(0, 255, 0), -1);
				if (dif > 12)
					max_ffg = 1;
			}
			//if (mres.size() ==  0)
			//	printf("process time: %fms\n", ptime);

			cv::imshow("result", dst_copy);
			for (int i = 0; i < pls.size(); i++)
			{
				vector <Point> ppnts;
				ppnts = pls[i];
				Point offset;
				offset = ppnts[0];
				offset.y -= 10;
				cv::putText(my_templ.roi_map, "ROI", offset, CV_FONT_HERSHEY_COMPLEX, 0.5, Scalar(255));
			}
			cv::imshow("roi_map", my_templ.roi_map);

			cv::waitKey(2);
			if (max_ffg)
			{
				waitKey();
			}
			if (kk > 355)
				kk = 0;
		}
	}
	cv::waitKey(0);

	return 1;
}