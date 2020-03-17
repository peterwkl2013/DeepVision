/*************************************************************************
*
*                    DeepVision SDK type defines
*
*
* File Discription:
*       type defines
*
* Author: WSN@2018/10/17
*
**************************************************************************/

#ifndef  __MVSDK_INTERFACE_H__
#define  __MVSDK_INTERFACE_H__

#include <opencv2/opencv.hpp>
#include <vector>
#include "mvSDK_type.h"


using namespace std;
using namespace cv;

typedef struct
{
	int index;                /* objext index */
	int label;                /* label value */
	int valid;                /* valid flag */
	Point center;             /* object center */
	float area;               /* object area */
	float angle;              /* object angle */
	float mean_gray;          /* object mean gray */
	float wh_reate;           /* w/h reate */
	float roundness;          /* roundness */
	Rect bound_rect;          /* object rect */
	RotatedRect box;          /* object box */
	Point2f box_pnts[4];      /* object box bounding rect */
	vector<Point> ppnts;      /* object points */
	vector<Point> contours;   /* object contours */
	int   reserved1;
	int   reserved2;
	float reserved3;
	float reserved4;
}mvCCLObject;

#ifdef __cplusplus
extern "C" {
#endif

extern int  mvImgRead(char *path, mvImage *imm, int color);
extern int	mvImageCreate(mvImage *pImg, int height, int width, int channel);
extern void mvImageDestroy(mvImage *pImg);
extern void mvImageShow(char *name, mvImage img);

#ifdef __cplusplus
}
#endif


/****************************** c++ function interface ******************************/

void*	 mvInstanceAlloc(int width, int height, algType type, initParam *para = NULL);

/***
server send message
 udp server ip: 127.0.0.1  port: 6090

@pdata   send data buffer
@lenght  data length
*/
int mvServerSendMsg(char *pdata, int length);

/***
transfrom a image by given below params.

@dat,    input, image data
@width,  input, image width
@height, input, image height
@nch,    input, image changel, support 1(grey), 3(rgb)
@hmat,   input, 3x3 perspective mat data[9]

@ hmat: 3x3 transform mat, eg:
1, 0, 1
0, 1, 1
0, 0, 1
return: MV_OK, channel error: -3
*/
int mvTransform2DImage(unsigned char *dat, int width, int height, int nch, double *hmat);

/***
transfrom a image by given below params.

@dat,    input, image
@hmat,   input, 3x3 perspective mat data[9]
*/
int mvTransorm2DImage(mvImage img, float *hmat);

/***
rotate one image by given the angle, scale.
anticlockwise mode

@img,    input, image changel, support 1(grey), 3(rgb)
@pos,    input, position point
@algle,  input, rotate
@scale,  input, the scale factor, eg scale = 1.0
*/
int mvRotate2DImage(mvImage img, mvPoint pos, float angle, float scale = 1.0);


/***
get the map point from other image[homography].

@inp,    input, the detect image point position
@hmat,   input, 3x3 perspective mat data[9]
@type,   input, type = 0,  calc the template image map point
type = 1,  calc the template image map point
@outp,   ouput, the detect image point position

*/
void mvMap2DPoint(mvFPoint inp, double *hmat, int type, mvFPoint &outp);


int mvSetDetRoiArea(void *ppAlg, mvDetRoi det, char *cfgpath);
int mvSetDetRoiArea(mvImage image, mvDetRoi det, char *filepath);
int mvSetDetRoiArea(mvDetRoi det, int width, int height, char *filepath);

/* c++ interface */
extern int mvSetDetRoiArea(void *ppAlg, mvDetRoi det, char *cfgpath);
extern int mvSetDetRoiArea(mvImage image, mvDetRoi det, char *filepath);
extern int mvSetDetRoiArea(mvDetRoi det, int width, int height, char *filepath);

/*
@image   input, 1 channel image
@det     input, det roi
@ouput,  image map
*/
extern int mvSetComponentRoiArea(mvImage& image, mvDetRoi& det);

/*
@roimap    input, 1 channel image
@edge      input, 1 channle image dat
ruturn   counter of map pixels and change the edge
*/
extern unsigned int mvDetRoiMapFilter(mvImage *roimap, unsigned char *edge);

/*
@det     input, det roi && roi_map=NULL or must to be set first
@ouput,  image map
ruturn   counter of map pixels and change the edge
*/
extern unsigned int mvDetRoiMapFilter(mvDetRoi det_roi, int width, int height, unsigned char *img);


extern int mvTempMatch(mvImage temp, mvImage det, mvRect &loc, vector<vector<mvPoint>> &ppt, int metho = 0);


/* point match && ouput the perspective matrix(3x3) store in matrix3x3
@ input,  tmplate point
@ input,  detect point
@ input,  thres = -1;  default thres
@ input,  max_ilter,  default: ilteration = 100
@ output, matrix3x3
@ output, angle
return true or false
*/

extern int mv2DPointMatch(vector<mvFPoint>& tmp, vector<mvFPoint>& det, double *matrix3x3, float& angle, float thres = -1, int max_ilter = 100);
extern int mv2DPointMatch(vector<vector<mvPoint>>& tmp, vector<vector<mvPoint>>& det, double *matrix3x3, float& angle, float thres = -1, int max_ilter = 100);
extern int mv2DPointMatch(vector<mvMatchPoint>& pair, double *matrix3x3, float& angle, float thres = -1, int max_ilter = 100);


/*
Draw the 3D coordinate axis
@input, image
@input, oxyz,  orignal, x, y, z point
@input, lenght, length default: 9
@input, thickness  default 2
return;
*/
extern void mvDraw3DCoordinateAxis(cv::Mat& image, vector<cv::Point>& oxyz, float lenth = 9, int thickness = 2);
extern void mvDraw3DCoordinateAxis(cv::Mat& image, vector<cv::Point2f>& oxyz, float lenth = 9, int thickness = 2);

extern void mvDraw3DCoordinateAxis(cv::Mat& image, vector<cv::Point>& oxyz, float angle, float lenth = 9, int thickness = 2);

/****************mv2d3dutils.h****************/
typedef struct
{
	vector<Point2f> tmp;
	vector<Point2f> dst;
}cvPairPoints;

/* extract contour point

@ input,   src_img input image
@ output,  detect contour point

return
*/
extern void mvExtractContour(Mat& src_img, vector<vector<mvPoint>> &ppt_det);
extern void mvExtractContour(Mat&  src_img, vector<vector<mvPoint>> &ppt_det);
extern void mvExtractContour(Mat&  src_img, vector<vector<mvPoint>> &ppt_det, Mat& loc_mask, int thres = 100);
extern void mvExtractContour(Mat& src_img, vector<Point2f>& ppt_det, Mat& loc_mask, int thres = 100);

extern void mvFillMask(vector<Point>& pnts, cv::Size dsize, Mat& mask_out);


/*
find image corners
dist_thres
filter points into two points-set.
@img,    input,  source image
@corners,  output,  corners of this image
@max_detect, input,  num corners to detect.
@use_subpix   input, if use subpix
@mask,  input,  mask
*/
extern void mvFindCorners(Mat& img, vector <Point2f>& corners, int max_detect, int use_subpix, Mat& mask);

/*
find pair points by Euclidean  distance.
dist_thres
filter points into two points-set.
@tmp,    input,  point set of first image
@dst,    input,  point set of the second image
@pairs   output, the pair point-set
@dist_thres,  input,  2 point distance threshold
@matrix3x3, input  the affine matrix 2x3
*/
extern void mvFindPairPoints(vector<Point2f>& tmp, vector<Point2f>& dst, cvPairPoints& pairs, float dist_thres, double* matrix3x3);

/*
filter points into two points-set.
@width,  input,  imgage width
@height, input,  imgage height
@tmp,    input,  point set of first the first image
@dst,    input,  point set of second the first image
@pairs   output, the pair point-set
@ilter,  input,  use the dilate metho to filt the two point set
@matrix3x3, input  the affine matrix 2x3

*/
extern void mvPointsFilter(int width, int height, vector<Point2f>& tmp, vector<Point2f>& dst, cvPairPoints& pairs, int ilter, double* matrix3x3);


/*
affine transition point to point

@input, temp   image
@input, center position of this temp
@input, dst center position of this dst image
@input, angle  rotaion angle
@input, scale  factor
@ouput, output transition point

return
*/
extern void mvCalAffinePoints(vector<vector<mvPoint>>& temp, Point temp_center, Point dst_center, float angle, float scale, vector<vector<mvPoint>>& dst);
extern void mvCalAffinePoints(vector<Point2f>& src, vector<Point2f>& dst, double* hmat);



/*
@label_img, input, label image  1 channel
@label_id,  input, label id which to featch the contours
@contours, output, contours output
@thres, input,  seach the smilar gray value threshold, default: 0
*/
extern void mvRegionGrow(Mat& label_img, uchar label_id, vector<vector<Point>>& contours, int thres = 0);

extern void mvRegionGrow(Mat& label_img, vector<vector<Point>>& contours, int thres);

extern void mvLabel2CCL(Mat label_img, Mat& gray_img, vector<mvCCLObject>& objs, int thres);


/***
component process

@img,    input, image data
@objs,   output object
@edge_thres,  input, edge detect threshold
@k1, input, dilate param
@k2, input, erode param
@label, input differrence threshold between the label image
*/
extern void mvComponentsProcess(Mat& img, vector<mvCCLObject>& objs, int edge_thres = 100, int k1 = 0, int k2 = 0, int label_thres = 0);


/***
component process, no roundness feature.

@img,    input, image data
@objs,   output object
@edge_thres,  input, edge detect threshold
@k1, input, dilate param
@k2, input, erode param
*/
extern void mvComponentsProcess2(Mat& img, vector<mvCCLObject>& objs, int edge_thres = 100, int k1 = 0, int k2 = 0);

/***
component extract feature
*/
extern void mvCCLExtractFeatures(Mat img, vector<mvCCLObject>& objs);


/***
circles detection.
@img,    input, image data
@objs,   output ccl object
@edge_thres,  input, edge detect threshold
@accumulator, input, accumulator threshold for the circle centers at the detection stage.
@min_radius, input, minimum circle radius. if 0, not use
@max_radius, input, maximum circles radius. if 0, not use
@kernel, input, image smooth kernal(3,5,7,9...)
*/
extern void mvCirclesDet(Mat& img, vector<mvCCLObject>& objs, int edge_thres = 100, int accu = 50,
	int min_radius = 0, int max_radius = 0, int kernel = 3);

#endif


 