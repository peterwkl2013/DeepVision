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

#include "DllmvInterface.h"
#include "mvSDK_interface.h"

using namespace cv;



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
}matchResult;

class mv2DTemplateMatch
{
public:
    mv2DTemplateMatch()
    {
        scale_ranges.push_back(1.0);
        angle_ranges.push_back(0.0); angle_ranges.push_back(360);
        num_features = 128; min_thres = 50; max_thres = 90;
        angle_step = 1;  filter_size = 5; /* */
        num_detect = 1;
        config_path = "d:/imvcfg/";
        overlap_thres = 0.7;
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
    @output, tf, output the template features
    */
    void getTemplateFeatures(String ids, vector<mvTemplate>&tf);

    /*
    @input, img, color image
    @input, ids, class name
    @input, offset, in the image
    @input, color
    */
    void drawTemplateFeatures(Mat& img, vector<mvTemplate>& tf, Point offset, Scalar color);

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
    @output, the hmat, get the template affine transform matrix.
    */
    void getAFTransform(matchResult& mres, double* hmat);

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
    int num_detect;                  /* num object to detect in one image */
    vector<float> scale_ranges;      /* scale factor(0.8~1.5) for training,  now only support scale = 1.0 */
    vector<float> angle_ranges;      /* rotation angle (0~360) for training,  default(0,360) */
    int   num_features;              /* feature to extract in the template */
    float min_thres;                 /* to control the feature response */
    float max_thres;                 /* to control the feature response */
    float   angle_step;              /* angle step when train template */
                                     /* when the template size(width, height) is too large  , it is necessary
                                     to set the filter size: 3, 5, 7, 9,11, to filter the features, else set to 0 , when have enough features */
    int   filter_size;
    double hmat[16];                /* output, current transformation matrix for : 2x3, 3x3, 4x4 */
    float reffine_angle;            /* output, the reffined angle of current transform */
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


static float radsToDegrees(float rads)
{
    return 180 * rads / 3.14159265358;
}

static float caclAngle(Point2f pnt, Point2f center)
{
    float angle = .0;
    float dx, dy;
    float dyx;

    dx = pnt.x - center.x;
    dy = pnt.y - center.y;

    angle = radsToDegrees(atan2(dy, dx));

    if (angle < 0)
        angle += 360;
    if (angle > 360)
        angle = 360;

    return (angle);
}


void  mvRefinePosition(Mat& tmp_gray, Mat& dst_gray, Point2f temp_center, Point2f current_center,
    float& current_angle, Point2f& refine_center, int angle_size, float angle_step,
    float thres, int xy_size, int winSize, double* matrix3x3)
{
    Mat img_roate;
    int xx, yy;
    //Mat tmp_copy,tmp_gray, dst_gray;
    Mat trans_img1, trans_img2;
    Mat trans_gray1, trans_gray2;
    Mat diff;
    int loc_x, loc_y;
    //int winSize = 32;
    double min_sumdiff = FLT_MAX;
    double best_angle;
    Point best_center;
    int index;
    int best_index = 0;

    int64 nTick, nTick2;
    double ptime;
    nTick = getTickCount();

    //cvtColor(dst, dst_gray, COLOR_BGR2GRAY);
    //cvtColor(temp, tmp_gray, COLOR_BGR2GRAY);


    index = 0;

    Mat trans_img;

    img_roate = cv::getRotationMatrix2D(current_center, current_angle, 1);

    warpAffine(tmp_gray, trans_img, img_roate, dst_gray.size());
    absdiff(trans_img, dst_gray, diff);
    imshow("rough", diff);
    //waitKey();

    for (float aa = -angle_size / 2.0; aa < angle_size / 2.0; aa += angle_step)
    {
        for (int hh = -xy_size / 2; hh < xy_size / 2; hh++)
        {//rows
            for (int ww = -xy_size / 2; ww < xy_size / 2; ww++)
            {//cols

                float this_angle = current_angle + aa;
                Point new_center = Point(cvRound(current_center.x + ww), cvRound(current_center.y + hh));
                img_roate = cv::getRotationMatrix2D(new_center, this_angle, 1);
                double *hmat = img_roate.ptr<double>();

                double sumdiff = .0;
                //Mat dst_copy, temp_copy;
                //dst_copy  = dst.clone();
                //temp_copy = temp.clone();
                //cv::circle(temp_copy, Point2f(temp_center.x, temp_center.y), 5, Scalar(0, 0, 255), -1);

                for (int rows = (temp_center.y - winSize / 2); rows < (temp_center.y + winSize / 2); rows++)
                {
                    for (int cols = (temp_center.x - winSize / 2); cols < (temp_center.x + winSize / 2); cols++)
                    {
                        int value;
                        loc_x = cvRound(hmat[0] * cols + hmat[1] * rows + hmat[2]);
                        loc_y = cvRound(hmat[3] * cols + hmat[4] * rows + hmat[5]);
                        if (loc_x < 0 || loc_x >= dst_gray.cols)
                            continue;
                        if (loc_y < 0 || loc_y >= dst_gray.rows)
                            continue;
                        double vv;
                        vv = fabs(*tmp_gray.ptr<uchar>(rows, cols) - *dst_gray.ptr<uchar>(loc_y, loc_x));
                        sumdiff += vv;
                        //cv::circle(temp_copy, Point2f(cols, rows), 1, Scalar(0, 255, 0), -1);
                        //cv::circle(dst_copy, Point2f(loc_x, loc_y), 1, Scalar(0, 255, 0), -1);
                        //imshow("temp_copy", temp_copy);
                        //imshow("dst_copy", dst_copy);
                        //waitKey();
                    }

                }

                //imshow("temp_copy", temp_copy);
                //imshow("dst_copy", dst_copy);

                sumdiff /= (winSize*winSize);
                //#pragma omp critical
                if (sumdiff < min_sumdiff)
                {
                    int ffg = 1;
                    min_sumdiff = sumdiff;
                    best_angle = this_angle;
                    best_center = new_center;
                    best_index = index;
                    cout << "index = " << index++ << ", cur=[" << sumdiff << "," << this_angle << "(" << new_center.x << "," << new_center.y << ")" << "]" <<
                        ",best=[" << min_sumdiff << "," << best_angle << "]" << "(" << best_center.x << "," << best_center.y << ")" <<
                        "------" << best_index << endl;

                    warpAffine(tmp_gray, trans_gray1, img_roate, dst_gray.size());
                    imshow("trans_gray1", trans_gray1);
                    absdiff(trans_gray1, dst_gray, diff);
                    imshow("test", diff);
                    threshold(diff, diff, 10, 255, THRESH_BINARY);
                    imshow("THRES", diff);
                    waitKey(20);
                    if (ffg)
                        waitKey();
                }
                //cout << "index = " << index++ << ", cur=[" << sumdiff <<","<< this_angle <<"("<<new_center.x<<","<<new_center.y<<")"<<"]"<<
                //	",best=[" << min_sumdiff <<","<<best_angle<<"]"<< "("<< best_center.x<<","<< best_center.y<<")"<<
                //	"------"<<best_index<<endl;

                //warpAffine(tmp_gray, trans_gray1, img_roate, dst_gray.size());
                //imshow("trans_gray1", trans_gray1);
                //absdiff(trans_gray1, dst_gray, diff);
                //imshow("test", diff);
                //waitKey(20);
                //winSize of compare image difference

                if (min_sumdiff < thres)
                    break;
            }

            if (min_sumdiff < thres)
                break;
        }

        if (min_sumdiff < thres)
            break;
    }
    waitKey();
    cout << "best:[" << best_angle << ",(" << best_center.x << "," << best_center.y << ")" << "]" << endl;
    img_roate = cv::getRotationMatrix2D(best_center, best_angle, 1);

    current_angle = best_angle;
    refine_center = best_center;

#ifndef MV_RELEASE
    cout << "best_matrix" << img_roate << endl;
#endif

#if 0
    warpAffine(tmp_gray, trans_gray1, img_roate, dst_gray.size());
    imshow("trans_gray1", trans_gray1);
    absdiff(trans_gray1, dst_gray, diff);
    imshow("best_diff", diff);
    waitKey();
#endif



    for (int ii = 0, index = 0; ii < img_roate.rows; ii++)
    {
        for (int cols = 0; cols < img_roate.cols; cols++)
        {
            double uptr = img_roate.at<double>(ii, cols);
            matrix3x3[index] = uptr;
            index++;
        }
    }

    nTick2 = getTickCount();
    ptime = ((double)nTick2 - nTick)*1000. / getTickFrequency();
    printf("reffine time: %fms\n", ptime);

    return;
}


void  mvRefinePosition2(Mat& tmp_gray, Mat& dst_gray, vector<Point2f> objs, Point2f& current_center,
    float& current_angle, int angle_size, float angle_step,
    int r_center, int win_size, int grid_size, double* matrix3x3)
{
    Mat img_roate;
    int xx, yy;
    //Mat tmp_copy,tmp_gray, dst_gray;
    Mat trans_img1, trans_img2;
    Mat trans_gray1, trans_gray2;
    Mat diff;
    //int winSize = 32;
    double min_sumdiff = FLT_MAX;
    double best_angle;
    Point best_center;
    int index;
    int best_index = 0;
    Point center;

    int64 nTick, nTick2;
    double ptime;
    nTick = getTickCount();

    vector<Point> rotate_pnts;
    vector<Point> temp_pnts;
    Point cc;
    cc.x = cc.y = 0;
    for (int ii = 0; ii < objs.size(); ii++)
    {
        //rotate_pnts.push_back(Point(objs[ii].x, objs[ii].y));
        cc.x += objs[ii].x;
        cc.y += objs[ii].y;
    }

    //
    center.x = cc.x / objs.size();
    center.y = cc.y / objs.size();
    //dst_pnts.push_back(center);

    for (int hh = -r_center / 2; hh < r_center / 2; hh++)
    {//rows
        for (int ww = -r_center / 2; ww < r_center / 2; ww++)
        {

            cc.x = cvRound(current_center.x + ww);
            cc.y = cvRound(current_center.y + hh);
            rotate_pnts.push_back(cc);
        }
    }

    for (int hh = -win_size / 2; hh < win_size / 2; hh++)
    {//rows
        for (int ww = -win_size / 2; ww < win_size / 2; ww++)
        {
            if (grid_size > 0 && (center.x + ww) % grid_size != 0)
                continue;

            cc.x = center.x + ww;
            cc.y = center.y + hh;
            temp_pnts.push_back(cc);
        }
    }

    index = 0;

    Mat trans_img;

#ifndef MV_RELEASE2
    img_roate = cv::getRotationMatrix2D(current_center, current_angle, 1);

    warpAffine(tmp_gray, trans_img, img_roate, dst_gray.size());
    absdiff(trans_img, dst_gray, diff);
    imshow("rough", diff);
    //waitKey();
#endif

    vector <float> angle_ranges;
    for (float aa = -angle_size / 2.0; aa < angle_size / 2.0; aa += angle_step)
    {
        angle_ranges.push_back(aa);
    }

    for (int kk = 0; kk < angle_ranges.size(); kk++)
    {

        //#pragma omp parallel for
        for (int ii = 0; ii < rotate_pnts.size(); ii++)
        {
            Mat temp_rotate(2, 3, CV_64F);
            double sumdiff;
            double *hmat;
            double this_angle = current_angle + angle_ranges[kk];
            Point new_center = rotate_pnts[ii];
            //#pragma omp critical
            {
                temp_rotate = cv::getRotationMatrix2D(new_center, this_angle, 1);
                //this_angle *= CV_PI / 180;

                //double alpha = std::cos(this_angle);
                //double beta = std::sin(this_angle);

                //double* m = temp_rotate.ptr<double>();

                //m[0] = alpha;
                //m[1] = beta;
                //m[2] = (1 - alpha)*new_center.x - beta * new_center.y;
                //m[3] = -beta;
                //m[4] = alpha;
                //m[5] = beta * new_center.x + (1 - alpha)*new_center.y;

            }
            hmat = temp_rotate.ptr<double>();

            //Mat dst_copy, temp_copy;
            //dst_copy  = dst.clone();
            //temp_copy = temp.clone();
            //cv::circle(temp_copy, Point2f(temp_center.x, temp_center.y), 5, Scalar(0, 0, 255), -1);
            sumdiff = .0;
            //#pragma omp parallel for reduction(+:sumdiff)
            for (int jj = 0; jj < temp_pnts.size(); jj++)
            {
                int value;
                int cols = temp_pnts[jj].x;
                int rows = temp_pnts[jj].y;
                int loc_x, loc_y;
                double vv;

                loc_x = cvRound(hmat[0] * cols + hmat[1] * rows + hmat[2]);
                loc_y = cvRound(hmat[3] * cols + hmat[4] * rows + hmat[5]);
                if (loc_x < 0 || loc_x >= dst_gray.cols)
                    continue;
                if (loc_y < 0 || loc_y >= dst_gray.rows)
                    continue;

                vv = fabs(*tmp_gray.ptr<uchar>(rows, cols) - *dst_gray.ptr<uchar>(loc_y, loc_x));
                sumdiff += vv;
                //cv::circle(temp_copy, Point2f(cols, rows), 1, Scalar(0, 255, 0), -1);
                //cv::circle(dst_copy, Point2f(loc_x, loc_y), 1, Scalar(0, 255, 0), -1);
                //imshow("temp_copy", temp_copy);
                //imshow("dst_copy", dst_copy);
                //waitKey();
            }

            //imshow("temp_copy", temp_copy);
            //imshow("dst_copy", dst_copy);

            sumdiff /= temp_pnts.size();
            //#pragma omp critical
#pragma omp critical
            if (sumdiff < min_sumdiff)
            {
                int ffg = 0;
                min_sumdiff = sumdiff;
                best_angle = this_angle;
                best_center = new_center;
#ifndef MV_RELEASE
                best_index = index;
                cout << "index = " << index++ << ", cur=[" << sumdiff << "," << this_angle << "(" << new_center.x << "," << new_center.y << ")" << "]" <<
                    ",best=[" << min_sumdiff << "," << best_angle << "]" << "(" << best_center.x << "," << best_center.y << ")" <<
                    "------" << best_index << endl;

                warpAffine(tmp_gray, trans_gray1, img_roate, dst_gray.size());
                imshow("trans_gray1", trans_gray1);
                absdiff(trans_gray1, dst_gray, diff);
                imshow("test", diff);
                threshold(diff, diff, 10, 255, THRESH_BINARY);
                imshow("THRES", diff);
                waitKey(20);
                if (ffg)
                    waitKey();
#endif

            }
            //cout << "index = " << index++ << ", cur=[" << sumdiff <<","<< this_angle <<"("<<new_center.x<<","<<new_center.y<<")"<<"]"<<
            //	",best=[" << min_sumdiff <<","<<best_angle<<"]"<< "("<< best_center.x<<","<< best_center.y<<")"<<
            //	"------"<<best_index<<endl;

            //warpAffine(tmp_gray, trans_gray1, img_roate, dst_gray.size());
            //imshow("trans_gray1", trans_gray1);
            //absdiff(trans_gray1, dst_gray, diff);
            //imshow("test", diff);
            //waitKey(20);
            //winSize of compare image difference
        }
    }
    printf("=====min_sumdiff = %lf\n", min_sumdiff);
    //waitKey();
    //cout << "best:[" << best_angle << ",(" << best_center.x << "," << best_center.y << ")" << "]" << endl;

    img_roate = cv::getRotationMatrix2D(best_center, best_angle, 1);

    current_angle = best_angle;
    current_center = best_center;

#ifndef MV_RELEASE
    cout << "best_matrix" << img_roate << endl;
#endif

#ifndef MV_RELEASE2
    warpAffine(tmp_gray, trans_gray1, img_roate, dst_gray.size());
    imshow("trans_gray1", trans_gray1);
    absdiff(trans_gray1, dst_gray, diff);
    imshow("best_diff000", diff);
    //waitKey();
#endif

    for (int ii = 0, index = 0; ii < img_roate.rows; ii++)
    {
        for (int cols = 0; cols < img_roate.cols; cols++)
        {
            double uptr = img_roate.at<double>(ii, cols);
            matrix3x3[index] = uptr;
            index++;
        }
    }

    nTick2 = getTickCount();
    ptime = ((double)nTick2 - nTick)*1000. / getTickFrequency();
    printf("reffine time: %fms\n", ptime);

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

    char *path1 = "E:/AlgDemo/3rdparty/pic/aoi/bd2_750x1024.bmp";
    char *path2 = "E:/AlgDemo/3rdparty/pic/aoi/bd2_750x1024.bmp";

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
    int trainFlag = 1;
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
    my_templ.angle_step = 0.2;
    //my_templ.scale_ranges.push_back(scl);

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

        my_templ.getTemplateFeatures("id1", tf);
        my_templ.drawTemplateFeatures(det_img1, tf, Point(offset[0].x, offset[0].y), Scalar(0, 255, 0));
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

        while (1)
        {
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

                my_templ.matchTemplate(det, class_ids, mres, conf);
#if 0
                my_templ.drawTemplateFeatures(det_img1, "id1", Scalar(0, 255, 0));
#else
                vector<mvTemplate > tf;

                my_templ.getTemplateFeatures("id1", tf);
                my_templ.drawTemplateFeatures(det_img1, tf, Point(offset[0].x, offset[0].y), Scalar(0, 255, 0));

#endif
                imshow("det_img1", det_img1);
                waitKey(10);

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


                for (int jj = 0; jj < mres.size(); jj++)
                {
                    matchResult obj = mres[jj];
                    vector <Point> pnts;
                    pnts.reserve(4);
                    pnts.push_back(Point(obj.bbox.pnts[0].x, obj.bbox.pnts[0].y));
                    pnts.push_back(Point(obj.bbox.pnts[1].x, obj.bbox.pnts[1].y));
                    pnts.push_back(Point(obj.bbox.pnts[2].x, obj.bbox.pnts[2].y));
                    pnts.push_back(Point(obj.bbox.pnts[3].x, obj.bbox.pnts[3].y));


                    /* draw location */
                    for (int i = 0; i < 4; i++)
                    {
                        int next = (i + 1 == 4) ? 0 : (i + 1);
                        cv::line(dst_copy, Point(obj.bbox.pnts[i].x, obj.bbox.pnts[i].y),
                            Point(obj.bbox.pnts[next].x, obj.bbox.pnts[next].y), Scalar(0, 0, 255), 2);
                        cv::String txt;
                        char str[512];

                        sprintf_s(str, "class_id:%s, score = %.2f, angle = %.1f(%.1f)", obj.class_id.c_str(), obj.score, my_templ.reffine_angle,
                            obj.angle);
                        Point temp_offset;
                        if (obj.class_id == "id1")
                            cv::putText(dst_copy, str, Point(10, 40), CV_FONT_HERSHEY_COMPLEX, 0.5, Scalar(0, 255, 0));
                        else
                            cv::putText(dst_copy, str, Point(10, 80), CV_FONT_HERSHEY_COMPLEX, 0.5, Scalar(0, 255, 0));

                        circle(dst_copy, Point(my_templ.rotation_center.x, my_templ.rotation_center.y), 3, Scalar(0, 0, 255), -1);
                    }
                }
            }
            cv::imshow("result", dst_copy);
            cv::waitKey(2);
        }
    }

    return 1;
}
