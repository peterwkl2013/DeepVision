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


#include <boost/archive/text_oarchive.hpp> 
#include <boost/archive/text_iarchive.hpp> 
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/serialization/string.hpp> 
#include <boost/serialization/export.hpp> 
#include <boost/serialization/list.hpp>
#include <boost/serialization/vector.hpp>

#include <iostream> 
#include <sstream> 
#include <string> 

struct mvTemplateFeature
{
	int x;         /* x offset */
	int y;         /* y offset */
	int label;     /* Quantization */
	mvTemplateFeature() : x(0), y(0), label(0) {}
	mvTemplateFeature(int _x, int _y, int _label) : x(_x), y(_y), label(_label) {};
};

typedef struct
{
	int width;
	int height;
	int x_offset;    /* feature offset x */
	int y_offset;    /* feature offset y */
	int pyramid_level;
	std::vector<mvTemplateFeature>features;
}mvTemplate;

typedef struct
{
	vector<float> scale_ranges;      /* scale factor(0.8~1.5) for training,  now only support scale = 1.0 */
	vector<float> angle_ranges;      /* rotation angle (0~360) for training,  default(0,360) */
	int total_id;          /*total template id */
	float angle_step;      /* angle step */
	int   padding;         /* padding size */
	Point2f tempInfo[4];   /* template location in tempalte image */
	Point2f objbox[4];     /* box of the template when angle = 0 */
	Point2f xy_offset;     /* template offset to the image */
	float width;           /* original width of template */
	float height;          /* original height of template */
	int num_features;      /* num features */
	int num_detected;      /* num have detect */
}TemplateExtraParam;

typedef std::map<String, TemplateExtraParam> TemplateExtraParamMap;
typedef std::vector<mvTemplate> TemplatePyramid;
typedef std::map<String, std::vector<TemplatePyramid>> TemplatesMap;

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
	String class_id;    /* class id */
	int index;          /* template index */
	float score;        /* score */
	float angle;        /* angle of this matched object*/
	float scale;        /* scale factor */
	Point2f r_center;   /* rotation center */
	Rect  rect;         /* rectangle location */
	mvFBox bbox;        /* box olcation */
	int x_offset, y_offset;
}matchResult;

class  mv2DTemplateMatch
{
public:
	mv2DTemplateMatch()
	{
		scale_ranges.push_back(1.0);
		angle_ranges.push_back(0.0); angle_ranges.push_back(360);
		sel_mode = 0;
		num_features = 128; min_thres = 50; max_thres = 90;
		angle_step = 1;  filter_size = 5; /* */
		num_detect = 1;
		config_path = "d:/imvcfg/";
		overlap_thres = 0.7;
		train_flg = 0;
	}
	mv2DTemplateMatch(vector<float>_scale,
		vector<float>_angle, int _num_features, int _min_thres, int _max_thres, float angle_step, int _filter_size,
		int _num_detect, char *_path) {
		scale_ranges = _scale;
		angle_ranges = _angle;
		num_features = _num_features; min_thres = _min_thres; max_thres = _max_thres;
		angle_step = angle_step;  filter_size = _filter_size;
		num_detect = _num_detect;
		config_path = _path;
		rotation_flag = 0;
	}

	~mv2DTemplateMatch();
	/*
	@input, temps  ,color-image
	@input, class_ids class name
	@input, template offset int the template iamge
	*/
	int trainTemplate(vector<Mat>& temps, vector<String>& class_ids, vector <Point2f> offset);
	void updateTemplates(void);
	/*
	@input,  ids, class name
	@input,  template index
	@output, tf, output the template features
	*/
	void getTemplateFeatures(String ids, int index, vector<mvTemplate>&tf);

	/*
	@input, img, color image
	@input, ids, class name
	@input, offset, in the image
	@input, color
	*/
	void drawTemplateFeatures(Mat& img, vector<mvTemplate>& tf, Point offset, Scalar color);

	/*
	@input, class ids
	@input, template features
	@output, features position points
	*/
	void getTemplateFeaturesPos(String ids, mvTemplate& tf, vector <Point2f>& outpoints);

	/*
	@input, img, color image
	@input, ids, class name
	@input, color
	*/
	void drawTemplateFeatures(Mat& img, String ids, Scalar color);

	/*
	@input, det_img  detection image,color-image
	@input, class_ids class name
	@input, conf  score confidence (0~100)
	@output, mres, match result
	@reutrn true or false
	*/
	int matchTemplate(Mat& det_img, vector<cv::String>& class_ids, vector<matchResult>& mres, float conf = 85);
	/*
	@output, the hmat, get the template affine transform matrix: 2x3.
	*/
	void getAFTransform(matchResult& mres, double* hmat);

	void refineObjectPos(Mat& temp, Mat& det, matchResult& mres, vector<Point2f>& mask_points, double* hmat);

	/*
		@output, the hmat, get the template homography transform matrix: 33.
	*/
	//void getHomoTransform(matchResult& mres, double* pmat);

	/*
	@temp,   input, template image,  gray image
	@dst,    input, detect image, gray image
	@objs    input, detect object position
	@angle_range, input, angle range(+/-), default: 5
	@angle_step,  input, angle step,  default: 0.2
	@r_center,    input, roation center offset range ,   default: 32
	@filt_size, the filter size within the windows size,  default: -1 (not use), 3, 5, 7,9...
	*/
	void refineRoughPosition(Mat& temp, Mat& dst, vector<Point2f> mask_points, int angle_range = 3, float angle_step = 0.2,
		int r_center = 32, int filt_size = -1);

	//void refineInternalPosition(Mat& temp, Mat& dst, vector<Point2f> mask_points);

	/*
	@tempp, input, template image points.
	@dstpp, input, detect image points, the size as the template.
	@type,  type = 0, the affine matrix: 2x3,  else the homnography matrix 3x3
	@matrix3x3, ouput, the affine matrix2x3 in [hmat]
	*/
	void refinePosition(vector<Point2f>&tempp, vector<Point2f>& dstpp, int type = 0);

	/*
	@tempp, input, template image points.
	@dstpp, input, detect image points, the size as the template.
	@max_lit,  ilteration, default: 50
	@thres,  ilteration threshold, default; 0.001
	@use_filter  set 1, to use the filter to improve the point matched accurrcy.
	*/
	void refinePositionICP(vector<Point2f>&tempp, vector<Point2f>& dstpp, int max_it = 50, float thres = 0.002, int use_filter = 0);

	/*
	@tempp, input, template points in the template image.
	@dstpp, output, output points in the detection image.
	*/
	void calcPointPosition(vector<Point2f>& tempp, vector<Point2f>& dstpp);


	//map(class_id , mvTemplate)
	std::map<String, std::vector<vector<mvTemplate>>> template_features;
	TemplateExtraParamMap class_param;

public:
	String config_path;              /* template feature to store or to read from */
	Mat det_img;                     /* detect image (input) */
	Mat temp_img;                    /* template image (input) */
	int sel_mode;                    /* select feature mode */
	int num_detect;                  /* num object to detect in one image */
	vector<float> scale_ranges;      /* scale factor(0.8~1.5) for training,  now only support scale = 1.0 */
	vector<float> angle_ranges;      /* rotation angle (0~360) for training,  default(0,360) */
	int   num_features;              /* feature to extract in the template */
	float min_thres;                 /* to control the feature response */
	float max_thres;                 /* to control the feature response */
	float   angle_step;              /* angle step when train template */
	int train_flg;                   /* after train new template or update template , be set to 1  */
	/* when the template size(width, height) is too large  , it is necessary
	to set the filter size: 3, 5, 7, 9,11, to filter the features, else set to 0 , when have enough features */
	int   filter_size;
	double hmat[16];                /* output, current transformation matrix for : 2x3, 3x3, 4x4 */
	float refine_angle;             /* output, the reffined angle of current transform */
	int rotation_flag;              /* check template is rotation flag */
	Point2f rotation_center;        /* template rotaion center */
	float overlap_thres;            /* multi-object with the same class id remove threshold,  rect overap */
private:
	int test;
};


extern void mvExtractContour(Mat& src_img, vector<vector<mvPoint>> &ppt_det);
//extern void mvExtractContour(Mat& src_img, vector<vector<mvPoint>> &ppt_det, Mat& loc_mask, int edge_thres = 100);
extern void mvCalAffinePoints(vector<vector<mvPoint>>& temp, Point temp_center, Point dst_center, float angle, float scale, vector<vector<mvPoint>>& dst);

extern float mv2DPointMatch_ICP(vector<Point2f>& tmp, vector<Point2f>& dst, double* matri3x3, double& angle,
	Mat& image, int max_ilter = 100, float _error = 0.001);

/* fill polygon area
@input, pnts input points
@input, image size,
@input,output, mask_out if empty create 3 channel mask image, else the input mask
return;
*/

extern void mvFillMask(vector<Point>& pnts, cv::Size dsize, Mat& mask_out);

class mvResStream
{
public:
	mvResStream()
	{
		frame_index = 0;
	}
	int frame_index;
	std::vector<String> id;
	std::vector<int> index;
	std::vector<float> score;
	std::vector<float> angle;
	std::vector<float> scale;
	std::vector<float> r_centerx;
	std::vector<float> r_centery;
	std::vector<float> rectx;
	std::vector<float> recty;
	std::vector<float> box_pnts1x;
	std::vector<float> box_pnts1y;
	std::vector<float> box_pnts2x;
	std::vector<float> box_pnts2y;
	std::vector<float> box_pnts3x;
	std::vector<float> box_pnts3y;
	std::vector<float> box_pnts4x;
	std::vector<float> box_pnts4y;
	std::vector<float> box_centerx;
	std::vector<float> box_centery;
	std::vector<float> box_angle;
	std::vector<float> box_width;
	std::vector<float> box_height;
	std::vector<float> box_area;
	std::vector<int> x_offset;
	std::vector<int> y_offset;

private:
	friend class boost::serialization::access;

	template <typename Archive>
	void serialize(Archive &ar, const unsigned int version)
	{
		ar & frame_index;
		ar & id;
		ar & index;
		ar & score;
		ar & angle;
		ar & scale;
		ar & r_centerx;
		ar & r_centery;
		ar & rectx;
		ar & recty;
		ar & box_pnts1x;
		ar & box_pnts1y;
		ar & box_pnts2x;
		ar & box_pnts2y;
		ar & box_pnts3x;
		ar & box_pnts3y;
		ar & box_pnts4x;
		ar & box_pnts4y;
		ar & box_centerx;
		ar & box_centery;
		ar & box_angle;
		ar & box_width;
		ar & box_height;
		ar & box_area;
		ar & x_offset;
		ar & y_offset;
	}
};
 

void testTransMatchResult(mvResStream& sres, vector <matchResult> & mres_out)
{
	matchResult obj;

	mres_out.clear();
	mres_out.resize(sres.id.size());

	for (int i = 0; i < sres.id.size(); i++)
	{
		mres_out[i].class_id = sres.id[i];
		mres_out[i].index = sres.index[i];
		mres_out[i].score = sres.score[i];
		mres_out[i].scale = sres.scale[i];
		mres_out[i].angle = sres.angle[i];
		mres_out[i].r_center.x = sres.r_centerx[i];
		mres_out[i].r_center.y = sres.r_centery[i];
		mres_out[i].rect.x = sres.rectx[i];
		mres_out[i].rect.y = sres.recty[i];
		mres_out[i].bbox.pnts[0].x = sres.box_pnts1x[i];
		mres_out[i].bbox.pnts[0].y = sres.box_pnts1y[i];
		mres_out[i].bbox.pnts[1].x = sres.box_pnts2x[i];
		mres_out[i].bbox.pnts[1].y = sres.box_pnts2y[i];
		mres_out[i].bbox.pnts[2].x = sres.box_pnts3x[i];
		mres_out[i].bbox.pnts[2].y = sres.box_pnts3y[i];
		mres_out[i].bbox.pnts[2].x = sres.box_pnts4x[i];
		mres_out[i].bbox.pnts[2].y = sres.box_pnts4y[i];
		mres_out[i].bbox.center.x = sres.box_centerx[i];
		mres_out[i].bbox.center.y = sres.box_centery[i];
		mres_out[i].bbox.angle = sres.box_angle[i];
		mres_out[i].bbox.area = sres.box_area[i];
		mres_out[i].bbox.width = sres.box_width[i];
		mres_out[i].bbox.height = sres.box_height[i];
		mres_out[i].x_offset = sres.x_offset[i];
		mres_out[i].y_offset = sres.y_offset[i];
	}

	return;
}

class  mvSerialPort
{

public:
	mvSerialPort()
	{
		connected = false;
	}
	~mvSerialPort();
	int openSerialPort(char *portName, int baud_rate);

	int readSerialPort(char *buffer, unsigned int buf_size);
	bool writeSerialPort(char *buffer, unsigned int buf_size);
	bool isConnected();
	void closeSerial();
private:
	void* handler;
	bool connected;
	int status;
	unsigned long errors;
};

/*
@res,    input, matchResult
@pdata,  output, string or binary data
@length, output, length size
@mode,   input, binary:mode = 0, binary: mode = 1
*/
extern void mvResultStreamOutput(vector<matchResult>& re, String& pdata, int& length, int mode = 0);


extern void mvTransMatchResult(mvResStream& sres, vector <matchResult> & mres_out);

static std::stringstream mvstream;
void testResultStreamOutput(vector<matchResult>& res, String& pdata, int& lenght)
{
	mvResStream val;
	static int mvframe_index;

	mvstream.clear();
	//boost::archive::text_oarchive oa(mvstream);
	boost::archive::binary_oarchive oa(mvstream);

	mvframe_index += 1;


	for (int i = 0; i < res.size(); i++)
	{
		matchResult obj = res[i];

		val.frame_index = mvframe_index;
		val.id.push_back(obj.class_id);
		val.index.push_back(obj.index);
		val.score.push_back(obj.score);
		val.angle.push_back(obj.angle);
		val.scale.push_back(obj.scale);
		val.r_centerx.push_back(obj.r_center.x);
		val.r_centery.push_back(obj.r_center.y);
		val.rectx.push_back(obj.rect.x);
		val.recty.push_back(obj.rect.y);
		val.box_pnts1x.push_back(obj.bbox.pnts[0].x);
		val.box_pnts1y.push_back(obj.bbox.pnts[0].y);
		val.box_pnts2x.push_back(obj.bbox.pnts[1].x);
		val.box_pnts2y.push_back(obj.bbox.pnts[1].y);
		val.box_pnts3x.push_back(obj.bbox.pnts[2].x);
		val.box_pnts3y.push_back(obj.bbox.pnts[2].y);
		val.box_pnts4x.push_back(obj.bbox.pnts[3].x);
		val.box_pnts4y.push_back(obj.bbox.pnts[3].y);
		val.box_centerx.push_back(obj.bbox.center.x);
		val.box_centery.push_back(obj.bbox.center.y);
		val.box_angle.push_back(obj.bbox.angle);
		val.box_width.push_back(obj.bbox.width);
		val.box_height.push_back(obj.bbox.height);
		val.box_area.push_back(obj.bbox.area);
		val.x_offset.push_back(obj.x_offset);
		val.y_offset.push_back(obj.y_offset);
	}

	//*pdata = (char*)mvstream.rdbuf();

	oa << val;
	pdata = mvstream.str();
	lenght = mvstream.str().length();

	return;
}

int main(int argc, char * argv[])
{


	int ret;
	mv2DTemplateMatch  my_templ;
	vector<matchResult>mres;

#if 1


	//char *path1 = "D:/pic/aoi/1_temp395x427.bmp";
	//char *path2 = "D:/pic/aoi/1.bmp";

	char *path1 = "D:/pic/aoi/bd2_750x1024.bmp";
	char *path2 = "D:/pic/aoi/bd2_750x1024.bmp";

	//char *path1 = "./pic/bd2_750x1024.bmp";
	//char *path2 = "./pic/bd2_750x1024.bmp";

#endif


	Mat rs;
	float scl = 1;

	Mat tmpl = imread(path1);
	//Mat tmpl2 = imread(path3);
	Mat det_img1 = imread(path2);

	resize(det_img1, rs, Size(det_img1.cols*scl, det_img1.rows*scl));
	det_img1 = rs.clone();
	tmpl = rs.clone();

	//Mat det_img2 = imread(path4);

	vector <Mat> img_dets, img_tmpls;
	vector<String> class_ids;

	String id1 = "0";
	String id2 = "1";
	int trainFlag = 0;
	int edge_thres = 90;
	int conf = 60;
	vector<vector<mvPoint>> ppt_det;
	vector<vector<mvPoint>> ppt_tmp, ppt_tmp2;
	vector<Point2f>pptf_tmp, pptf_trans, pptf_dst;

	Mat mask_out;
	Point temp_offset1(400 * scl, 400 * scl);
	Point temp_offset2(200 * scl, 400 * scl);
	Point temp_offset;
	double matrix4x4[16];
	double matrix3x3[9];

	double matrix3x3b[9];

	float angle;
	Mat trans_img, trans_img2;
	Mat hmat = Mat(Size(3, 3), CV_64FC1, (void*)matrix3x3);
	Mat diff;
	vector <Point2f> offset;
	double ptime;
	int64 nTick, nTick2;

#if 1

	Mat tmp_roi = tmpl(Rect(temp_offset1.x, temp_offset1.y, 200 * scl, 200 * scl));

	offset.push_back(temp_offset1);
	img_tmpls.push_back(tmp_roi);
	class_ids.push_back("id1");


	//Mat tmp_roi2 = tmpl(Rect(temp_offset2.x, temp_offset2.y, 160, 200));
	//offset.push_back(temp_offset2);

	//img_tmpls.push_back(tmp_roi2);
	//class_ids.push_back("id2");


	img_dets.push_back(det_img1);
#else
	img_dets.push_back(det_img1);
	img_dets.push_back(det_img2);

	img_tmpls.push_back(tmpl);
	img_tmpls.push_back(tmpl2);


	class_ids.push_back(id1);
	class_ids.push_back(id2);
#endif

	my_templ.num_detect = 1;
	my_templ.filter_size = 3;
	my_templ.num_features = 64;
	my_templ.angle_step = 1;
	my_templ.max_thres = 60;
	my_templ.sel_mode = 0;
	if (trainFlag)
	{
		cout << "train template..." << endl;
		ret = my_templ.trainTemplate(img_tmpls, class_ids, offset);
		if (ret <= 0)
			cout << "train template error! please change \
			the image size，filter_size or num_features size" << endl;

#if 0
		my_templ.drawTemplateFeatures(det_img1, "id1", Scalar(0, 255, 0));
#else
		vector<mvTemplate > tf;

		my_templ.getTemplateFeatures("id1", 0, tf);
		
		TemplateExtraParam t_param = my_templ.class_param["id1"];;
		
		//int xx, yy;
		//xx = t_param.padding / 2 + offset[0].x - tf[0].x_offset;
		//yy = t_param.padding / 2 + offset[0].y - tf[0].y_offset;
		vector<Point2f>  feature_points;
		my_templ.getTemplateFeaturesPos("id1", tf[0], feature_points);
		for (Point2f point : feature_points)
		{
			circle(det_img1, point,2, Scalar(255, 0, 0));
		}
		my_templ.drawTemplateFeatures(det_img1, tf, Point(t_param.xy_offset.x, t_param.xy_offset.y), Scalar(0, 255, 0));
		rectangle(det_img1, Rect(offset[0].x, offset[0].y, img_tmpls[0].cols, img_tmpls[0].rows), Scalar(0, 0, 255));
		//my_templ.getTemplateFeatures("id1",  tf);
		//my_templ.drawTemplateFeatures(det_img1, tf, Point(offset[0].x, offset[0].y), Scalar(0, 255, 0));
#endif
		imshow("draw_features", det_img1);
		waitKey();
	}


	vector<Point2f>corners_tmp, corners_dst, filter_out;
	Mat cmask;
	vector <Point> mpnt;
	mpnt.push_back(Point(temp_offset1.x, temp_offset1.y));
	mpnt.push_back(Point(temp_offset1.x + img_tmpls[0].cols, temp_offset1.y));
	mpnt.push_back(Point(temp_offset1.x + img_tmpls[0].cols, temp_offset1.y + img_tmpls[0].rows));
	mpnt.push_back(Point(temp_offset1.x, temp_offset1.y + img_tmpls[0].rows));

	mvFillMask(mpnt, tmpl.size(), cmask);
	//imshow("mask", cmask);
	//waitKey();

	Mat corner = tmpl.clone();

#ifdef USE_CORNERS
	mvFindCorners(corner, corners_tmp, 10, 1, cmask);
#else
	mvExtractContour(corner, corners_tmp, cmask, edge_thres);
#endif
	imshow("corners_tmp", corner);
	//waitKey();

	for (size_t ii = 0; ii < img_dets.size(); ii++)
	{


		Mat det_img = img_dets[ii];


		Point2f center(det_img.cols / 2, det_img.rows / 2);
		Mat img_roate;
		Mat det, det_affine;
		Mat tmp_copy, tmp_gray;
		Mat dst_copy, dst_gray;


		for (float kk = 0; kk < 359; kk += 3)
		{
			//for (int ee = 0; ee < 100; ee += 10)

			Point2f oo = Point(100, 100);
			//if (kk == 0)
			//{
			//	//定义平移矩阵
			//	cv::Mat t_mat = cv::Mat::zeros(2, 3, CV_32FC1);

			//	t_mat.at<float>(0, 0) = 1;
			//	t_mat.at<float>(0, 2) = oo.x; //水平平移量
			//	t_mat.at<float>(1, 1) = 1;
			//	t_mat.at<float>(1, 2) = oo.y; //竖直平移量


			//	//img_roate = cv::getRotationMatrix2D(center + oo, kk, 1);
			//	////det_clone = det_img.clone();
			//	//det_img = img_dets[ii].clone();
			//	//circle(det_img, Point(200, 400), 10, Scalar(0, 0, 0), -1);
			//	cv::warpAffine(det_img, det, t_mat, det_img.size());
			//	imshow("det", det);
			//	//waitKey();


			//}
			//else
			{
				img_roate = cv::getRotationMatrix2D(center + oo, kk, 1);
				//det_clone = det_img.clone();
				det_img = img_dets[ii].clone();
				circle(det_img, Point(200, 400), 10, Scalar(0, 0, 0), -1);
				det = img_dets[ii].clone();
				circle(det, Point(200, 400), 10, Scalar(0, 0, 0), -1);
				cv::warpAffine(det_img, det, img_roate, det_img.size());
			}

			//imshow("ccc", det);
			//waitKey();
			nTick2 = getTickCount();
			my_templ.matchTemplate(det, class_ids, mres, conf);
			nTick2 = getTickCount();
			ptime = ((double)nTick2 - nTick)*1000. / getTickFrequency();
			printf("match time: %fms\n", ptime);

			String pdata = "hello xvision\n";
			char ppdat[4096];
			int length;
			static mvSerialPort mycom;
			static int mode = 0;
			std::stringstream mvstream;

			mvstream.str("");  //clear()
			/* 结果序列化：将mres结果转成序列化流 */
			mvResultStreamOutput(mres, pdata, length, mode);
			//mvResultStreamOutput(mres, ppdat, length);
			//testResultStreamOutput(mres, pdata, length);

			/* 将序列化流从服务器网络传送, port:6090, send "xvision" first */
			int ret = mvServerSendMsg((char*)pdata.data(), length);

			/* 将序列化流串口RS232传送*/
			if (mycom.isConnected() == false)
			{
				mycom.openSerialPort("COM4", 115200);
			}
			else
			{
				mycom.writeSerialPort((char*)pdata.data(), length);
			}

			/* 结果反序列化：序列化流结果转成mytres结果*/
			mvResStream pp;
			//std::ifstream iss(pdata.c_str(), std::ios::binary);

			if (mode == 0)
			{ 
				mvstream.write(pdata.data(), length);

				boost::archive::binary_iarchive ia(mvstream);
				ia >> pp;
			}
			else if (mode == 1)
			{
				mvstream << pdata << endl;
				boost::archive::text_iarchive ia(mvstream);
				ia >> pp;
				std::cout << mvstream.str() << endl;
			}
						
			vector<matchResult>mytres;
			mvTransMatchResult(pp, mytres);
			//testTransMatchResult(pp, mytres);

			static int ccflag = 0;
			if (ccflag == 0)
			{
				vector<mvTemplate > tf;

				my_templ.getTemplateFeatures("id1", 0, tf);

				TemplateExtraParam t_param = my_templ.class_param["id1"];;
				imshow("draw_features0", det_img1);
				waitKey(2);
				ccflag = 1;

			}

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


			for (int jj = 0; jj < mytres.size(); jj++)
			{
				matchResult obj = mres[jj];
				vector <Point> pnts;

				pnts.reserve(4);
				pnts.push_back(Point(obj.bbox.pnts[0].x, obj.bbox.pnts[0].y));
				pnts.push_back(Point(obj.bbox.pnts[1].x, obj.bbox.pnts[1].y));
				pnts.push_back(Point(obj.bbox.pnts[2].x, obj.bbox.pnts[2].y));
				pnts.push_back(Point(obj.bbox.pnts[3].x, obj.bbox.pnts[3].y));
				std::vector<mvTemplate > template_feature;
				vector<Point2f>  feature_points;

				/* 获取当前目标的特征点，并绘制特征点 */
				my_templ.getTemplateFeatures(obj.class_id, obj.index, template_feature);
				//my_templ.getTemplateFeaturesPos(obj.class_id, template_feature[0], feature_points);
		
				for (int i = 0; i < template_feature[0].features.size(); i++)
				{
					int xx, yy;
					xx = template_feature[0].features[i].x + obj.x_offset;
					yy = template_feature[0].features[i].y + obj.y_offset;

					circle(dst_copy, Point2f(xx,yy), 3, Scalar(255, 0, 0), -1);
				}

				my_templ.getAFTransform(obj, matrix3x3);
				Mat affine_mat = Mat(Size(3, 2), CV_64FC1, (void*)matrix3x3);
				Point2f obj_center;
				
				/* draw location */
				for (int i = 0; i < 4; i++)
				{
					int next = (i + 1 == 4) ? 0 : (i + 1);
					cv::line(dst_copy, Point(obj.bbox.pnts[i].x, obj.bbox.pnts[i].y),
						Point(obj.bbox.pnts[next].x, obj.bbox.pnts[next].y), Scalar(0, 0, 255), 2);
					cv::String txt;
					char str[512];

				/*	sprintf(str, "class_id:%s, score = %.2f, angle = %.1f(%.1f)", obj.class_id.c_str(), obj.score, my_templ.reffine_angle,
						obj.angle);*/
					Point temp_offset;
					if (obj.class_id == "id1")
						cv::putText(dst_copy, str, Point(10, 40), CV_FONT_HERSHEY_COMPLEX, 0.5, Scalar(0, 255, 0));
					else
						cv::putText(dst_copy, str, Point(10, 80), CV_FONT_HERSHEY_COMPLEX, 0.5, Scalar(0, 255, 0));
					obj_center = (obj.bbox.pnts[0] + obj.bbox.pnts[1] + obj.bbox.pnts[2] + obj.bbox.pnts[3]) / 4;
					circle(dst_copy, obj_center, 3, Scalar(0, 0, 255), -1);
					line(dst_copy, cvPoint(obj_center.x, obj_center.y - 20),
						cvPoint(obj_center.x, obj_center.y + 20), CV_RGB(0, 0, 255), 1, CV_AA, 0);
					line(dst_copy, cvPoint(obj_center.x - 20, obj_center.y), cvPoint(obj_center.x + 20,
						obj_center.y), CV_RGB(0, 0, 255), 1, CV_AA, 0);;
				}

			    /* 获取index = 0 时的特征点*/
				my_templ.getTemplateFeatures(obj.class_id, 0, template_feature);
				my_templ.getTemplateFeaturesPos(obj.class_id, template_feature[0], feature_points);
				/* 目标姿态调整 */

				nTick = getTickCount();
				my_templ.refineObjectPos(tmp_gray, dst_gray, obj, feature_points, matrix3x3);
				nTick2 = getTickCount();
				ptime = ((double)nTick2 - nTick)*1000. / getTickFrequency();
				printf("refine time: %fms\n", ptime);
				for (int i = 0; i < 4; i++)
				{
					int next = (i + 1 == 4) ? 0 : (i + 1);
					cv::line(dst_copy, Point(obj.bbox.pnts[i].x, obj.bbox.pnts[i].y),
						Point(obj.bbox.pnts[next].x, obj.bbox.pnts[next].y), Scalar(255, 0, 0), 2);
					cv::String txt;
					char str[512];

					sprintf(str, "class_id:%s, score = %.2f, angle = %.1f(%.1f)", obj.class_id.c_str(), obj.score, my_templ.refine_angle,
						obj.angle);
					Point temp_offset;
					if (obj.class_id == "id1")
						cv::putText(dst_copy, str, Point(10, 40), CV_FONT_HERSHEY_COMPLEX, 0.5, Scalar(0, 255, 0));
					else
						cv::putText(dst_copy, str, Point(10, 80), CV_FONT_HERSHEY_COMPLEX, 0.5, Scalar(0, 255, 0));
					obj_center = (obj.bbox.pnts[0] + obj.bbox.pnts[1] + obj.bbox.pnts[2] + obj.bbox.pnts[3]) / 4;
					circle(dst_copy, obj_center, 3, Scalar(0, 255, 255), -1);
					line(dst_copy, cvPoint(obj_center.x, obj_center.y - 20), 
						cvPoint(obj_center.x, obj_center.y + 20), CV_RGB(0, 255, 255), 1, CV_AA, 0);
					line(dst_copy, cvPoint(obj_center.x -20, obj_center.y), cvPoint(obj_center.x + 20, 
						obj_center.y), CV_RGB(0, 255, 255), 1, CV_AA, 0);
				}

			}
			cv::imshow("result", dst_copy);
			cv::waitKey(2);
			if (kk + 5 > 357)
				kk = 0;
		}
	}

	return 1;
}
