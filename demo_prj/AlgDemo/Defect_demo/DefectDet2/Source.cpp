/************************************************************************
* File      :  demo.cpp
*
* 旋转和缺陷定位检测demo
*
* By Wsn
* 2017-OCT-27
************************************************************************/
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
using namespace std;

static RNG rng(12345);

algDllHandle *pAlg = NULL;


//alg算法配置
//算法缩放系数
#define MV_ALG_SCL     1
//算法配件路径
#if 1
#define MV_CONFIG_PATH   "E:/AlgDemo/3rdparty/imvcfg/"
#define TEST_SAMPLE_PATH "E:/AlgDemo/3rdparty/pic/aoi/"
#define TEST_PATH "imageList_temp.txt"
#else
#define MV_CONFIG_PATH   "./imvcfg/"
#define TEST_SAMPLE_PATH "./pic/"
#define TEST_PATH "imageList.txt"
#endif

//输入图像缩放系数
#define MV_INTPUT_IMAGE_SCAL   1
//算法类型
#define MV_ALG_TYPE  (MV_ALG_DEFECTSDET2)

typedef struct defectsFeature
{/* match Objects */
    int   index;           /* object index */
    int   mat_index;       /* matched object index */
    int   valid;           /* valid flag */
    int   num_pixels;
    float mean_gray;
    float area;
    float wh_rate;
    float angle;
    float contrast;
    float homogeneity;
    float entropy;
    float energy;
    float correlation;
    int   reserved1;
    float reseved2;
}defectsFeature;

typedef struct
{
    mvRGBImage tmp_img;                         /* template imgage */
    mvRGBImage cur_img;                         /* current process imgage */
    double     hmat[9];                         /* 3x3 transform mat */
    float      angle;                           /* image rotation angle */
    std::vector<defectsFeature> matObjFeature;  /* defects feature */
    std::vector<cv::Point2f> temp_points;       /* template edge points */
}defectsUserResult;

/***
draw mvCCLItems

*/
void cclTtemsDraw(mvCCLItems *pComps, cv::Mat img)
{
    int i, j, ind;
    int tmpx, tmpy;
    int ww, hh;
    int flag;
    mvCCLItem *pComp;
    mvRect rc;
    mvPoint pt;
    int x, xx, y, yy;
    cv::Point pp1, pp2;
    vector<mvFPoint> subpixels;

    for (i = pComps->num_comp - 1; i >= 0; i--)
    {
        cv::Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));

        pComp = &pComps->ccomps[i];
        if (!pComp->vflag)
            continue;
        x = pComp->rec_bound.up_left.x;
        y = pComp->rec_bound.up_left.y;
        xx = pComp->rec_bound.dw_right.x;
        yy = pComp->rec_bound.dw_right.y;

        rc = pComp->rec_bound;
        pt = pComp->pnt_mass;
        //mvImageDrawLine(pImage, pt, pt, 255, 1);
        //mvImageDrawRectangle(pImage, 255, rc.up_left.y, rc.dw_right.y,
        //  rc.up_left.x, rc.dw_right.x);
        cv::rectangle(img, cv::Rect(x, y, xx - x, yy - y), Scalar(0, 0, 255));
        for (int k = 0; k < pComp->pnt_countour.nump; k++)
        {
            //mvLog("maxOds:%f,", pods[k]);
            pp1.x = pComp->pnt_countour.pnts[k].x;
            pp1.y = pComp->pnt_countour.pnts[k].y;

            cv::circle(img, pp1, 1, color, 1, 8, 0);
        }
        //draw "+"
        int len = 10;
        pp1.x = pComp->pnt_center.x;
        pp1.y = pComp->pnt_center.y - len;
        pp2.x = pComp->pnt_center.x;
        pp2.y = pComp->pnt_center.y + len;
        rectangle(img, pp1, pp2, CV_RGB(0, 255, 0));
        pp1.x = pComp->pnt_center.x - len;
        pp1.y = pComp->pnt_center.y;
        pp2.x = pComp->pnt_center.x + len;
        pp2.y = pComp->pnt_center.y;
        rectangle(img, pp1, pp2, CV_RGB(0, 255, 0));
        /*object centor */
        pp1.x = pComp->pnt_center.x, pp1.y = pComp->pnt_center.y;
        line(img, pp1, pp1, cvScalar(0, 0, 255), 2);
    }

    imshow("ccls", img);

    //mvSubpixelsExtract(pComps->pnts_bank, pComps->width, pComps->height, subpixels);

    return;
}


//opencv interface
extern int mvTempMatch(Mat temp_img, Mat det_img, mvRect &loc, vector<vector<mvPoint>> &ppt, int metho = 0);
extern void mvExtractContour(Mat src_img, vector<vector<mvPoint>> &ppt_det);

extern int mvImageMatch(Mat& tmp, Mat& dst, vector<mvMatchPoint>& pair, double *matrix3x3, float *angle, int point_thres = 11);

extern int mvExtractKeypoints(Mat& tmp, Mat& dst, vector<mvMatchPoint>& pair, float ratio_thresh = 0.75f, int patch_size = 11);


/*
location two image with ransac metho to remove the mistake points,
the more the matched pair, the accuracy maybe better.

@input,  tmp, template image
@input,  dst, detection image
@input,  dist_thres distance threshold to fecth the feature point
@input,  patch_size threshold to fecth the feature point
@output  matched feature point
@output  translation matrix3x3 && angle

*/
extern int mvExtractKeypointsWithRansac(Mat& tmp, Mat& dst, vector<mvMatchPoint>& pair, double *matrix3x3, float *angle, float dist_thres = 0.7f, int patch_size = 11);


int main(int argc, char* argv[])
{
    int key, ret;
    initParam param;
    mvResult *pRes;
    defectsUserResult *eval;
    vector<mvFPoint>cc1, cc2;

    char *path1 = "e:/AlgDemo/3rdparty/pic/aoi/bd2_750x1024.bmp";
    char *path2 = "e:/AlgDemo/3rdparty/pic/aoi/bd2_750x1024b.bmp";


    //char *path1 = "d:/pic/aoi/1.bmp";
    //char *path2 = "d:/pic/aoi/2.bmp";


    //char *path1 = "d:/pic/aoi/bd2_temp.bmp";
    //char *path2 = "d:/pic/aoi/bd2_tempb436x270.bmp";

    //char *path1 = "d:/pic/aoi/2_temp446x336.bmp";
    //char *path2 = "d:/pic/aoi/2_temp446x336b.bmp";

    //mvPCL_ICP(cc1, cc2);
    Mat tmp, dst;
    double matrix3x3[16]{ 0 };

    float angle;

    tmp = imread(path1);
    dst = imread(path2);

    //cv::resize(tmp, tmp, cv::Size(tmp.cols * 0.25, tmp.rows*0.25));
    //cv::resize(dst, dst, cv::Size(dst.cols * 0.25, dst.rows*0.25));

    imshow("tmp", tmp);
    imshow("dst", dst);
    vector<mvMatchPoint>match;
    Mat trans_mat, trans1;
    ////方法一, 求出转换矩阵，输出给缺陷检测算子
    mvExtractKeypointsWithRansac(tmp, dst, match, matrix3x3, &angle, 0.75, 11);

    //再次精匹配求出matrix3x3
    mv2DPointMatch(match, matrix3x3, angle);
    //trans_mat = Mat(3, 3, CV_64FC1, (void*)matrix3x3);
    //cout << "M=" << trans_mat << endl;
    //cout << "angle=" << angle << endl;
    //warpPerspective(tmp, trans1, trans_mat, dst.size(), INTER_LINEAR, BORDER_REPLICATE);
    //imshow("tran1b0", trans1);

    ////方法二，求出转换矩阵，输出给缺陷检测算子
    //mvExtractKeypoints(tmp, dst, match, 0.75, 11);

    //mv2DPointMatch(match, matrix3x3, angle);

    //trans_mat = Mat(3, 3, CV_64FC1, (void*)matrix3x3);
    //cout << "M=" << trans_mat << endl;
    //cout << "angle=" << angle << endl;
    //warpPerspective(tmp, trans1, trans_mat, dst.size(), INTER_LINEAR, BORDER_REPLICATE);
    //imshow("tran1b1", trans1);
    //waitKey();

    //设置alg param初始化参数
    strcpy_s(param.cfg_path, MV_CONFIG_PATH);
    //param.tmp_offset        = offset;
    //param.tmp_roi           = loc;
    param.online_temp = 0;    //not used
    param.tmp_img.pframe = NULL;
    param.tmp_img.width = 0;        //temp->width;
    param.tmp_img.height = 0;       //temp->height;
    param.tmp_img.channels = 0;     //temp->nChannels;
    param.tmp_img.depth = 0;        //temp->depth;
    param.tmp_img.wstep = 0;    //temp->wstep;
    param.tmp_img.type = MV_RGB24;        //MV_CV_IPL;



    pAlg = (algDllHandle*)mvInstanceAlloc(dst.cols, dst.rows, MV_ALG_DEFECTSDET2, &param);
    if (pAlg == NULL)
        return -1;

    pRes = (mvResult*)&pAlg->result;


    int temp_flag = 0;
    mvInputImage orgImage;
    mvTempImg temp;

    //process loop
    {

        if (temp_flag == 0)
        {
            orgImage.index = 0;       /* index = 0,  template, */
            orgImage.pframe = (void*)tmp.data;        /* 图像数据地址 */
            orgImage.width = tmp.cols;                /* 图像宽度 */
            orgImage.height = tmp.rows;               /* 图像高度 */
            orgImage.channels = tmp.channels();       /* 图像通道*/
            orgImage.wstep = tmp.cols * dst.channels();  /* 图像 wstep = 宽度* 通道数 */
            orgImage.nsize = tmp.cols * dst.rows * dst.channels();
            orgImage.depth = tmp.depth();         /* 图像深度 */
            orgImage.type = MV_BGR24;   /*帧的格式*/
            ret = mvAlgProcess(pAlg, (mvInputImage*)&orgImage);

            temp_flag = 1;
            //waitKey();
        }

        orgImage.index = 1;           /* index = 0,  template, */
        orgImage.pframe = (void*)dst.data;        /* 图像数据地址 */
        orgImage.width = dst.cols;                /* 图像宽度 */
        orgImage.height = dst.rows;               /* 图像高度 */
        orgImage.channels = dst.channels();       /* 图像通道*/
        orgImage.wstep = dst.cols * dst.channels();  /* 图像 wstep = 宽度* 通道数 */
        orgImage.nsize = dst.cols * dst.rows * dst.channels();
        orgImage.depth = dst.depth();         /* 图像深度 */
        orgImage.type = MV_BGR24;   /*帧的格式*/

                                    /* 把转换矩阵数据传入*/
        temp.temp.index = 0;
        temp.temp.pframe = (void*)tmp.data;
        temp.temp.width = tmp.cols;
        temp.temp.height = tmp.rows;
        temp.temp.channels = tmp.channels();
        orgImage.nsize = tmp.cols * dst.rows*dst.channels();
        temp.temp.wstep = tmp.cols * dst.channels();
        temp.temp.depth = tmp.depth();
        temp.temp.type = MV_BGR24;
        temp.type = 0;
        /* 传递变换matrix数据*/
        for (int i = 0; i < 9; i++)
            temp.tran_matrix[i] = matrix3x3[i];

        orgImage.user_param = (void*)&temp;

        ret = mvAlgProcess(pAlg, (mvInputImage*)&orgImage);
        eval = (defectsUserResult*)pAlg->result.user_dat;

        Mat input2 = dst.clone();

        if (ret)
        {
            cv::Mat tmp2;
            char sss[80];

            if (eval->tmp_img.channels == 1)
                tmp2 = Mat(eval->tmp_img.height, eval->tmp_img.width, CV_8UC1, (void*)eval->tmp_img.pdata);
            else if (eval->tmp_img.channels == 3)
                tmp2 = Mat(eval->tmp_img.height, eval->tmp_img.width, CV_8UC3, (void*)eval->tmp_img.pdata);
            //imshow("tmp-img", tmp2);

            /* 算法处理结果输出 */
            sprintf_s(sss, "angle = %.4f", eval->angle);
            cout << sss << endl;
            cout << "==============object feature==============" << endl;
            for (int kk = 0; kk < pRes->mat_objs.num_obj; kk++)
            {
                matchObj matobj;
                defectsFeature fea;

                matobj = pRes->mat_objs.mat_obj[kk];
                fea = eval->matObjFeature[kk];
                /*cout << "index=" << matobj.mat_index << endl;
                cout << "num_pixels=" << fea.num_pixels << endl;
                cout << "area=" << fea.area << endl;
                cout << "mean_gray=" << fea.mean_gray << endl;
                cout << "angle=" << fea.angle << endl;
                cout << "contrast=" << fea.contrast << endl;
                cout << "homogeneity=" << fea.homogeneity << endl;
                cout << "entropy=" << fea.entropy << endl;
                cout << "correlation=" << fea.correlation << endl;
                cout << "wh_rate=" << fea.wh_rate << endl;*/
            }
            cout << "==============object feature==============" << endl;
            //绘制defects检测结果
            mvMatchObjsDrawAndDisplay(pAlg, pRes);
            //绘制组件
            //cclTtemsDraw(pRes->cc_items, dst);
            //imwrite("ccls.png", input2);
            //显示defects检测结果
            imshow("det-result", dst);
            imwrite("det-result.png", dst);

            {
                int font_face = cv::FONT_HERSHEY_SIMPLEX;
                mvFPoint mm, om1, om2;

                //imshow("org-img", input2);
                //标记点选择
                mm.x = input2.cols / 2;  mm.y = input2.rows / 2;
                cv::circle(input2, cvPoint(mm.x, mm.y), 3, cv::Scalar(0, 255, 0));
                cv::circle(tmp, cvPoint(mm.x, mm.y), 2, cv::Scalar(0, 255, 255));

                //计算模板图中坐标om1
                mvMap2DPoint(mm, eval->hmat, 1, om1);
                cv::circle(tmp, cvPoint(om1.x, om1.y), 3, cv::Scalar(0, 255, 0));
                //计算检测图中坐标om2
                mvMap2DPoint(mm, eval->hmat, 0, om2);
                cv::circle(input2, cvPoint(om2.x, om2.y), 3, cv::Scalar(0, 255, 255));

                //imshow("tmp-img-map1", tmp);
                //imshow("org-img-map1", input2);

                //合并图像
                cv::Mat merge, left, right;
                cv::Size ss((tmp.cols + input2.cols), max(tmp.rows, input2.rows));
                merge.create(ss, CV_MAKETYPE(CV_8U, 3));

                left = merge(cv::Rect(0, 0, tmp.cols, tmp.rows));
                tmp.copyTo(left);

                right = merge(cv::Rect(tmp.cols, 0, input2.cols, input2.rows));
                input2.copyTo(right);
                //imshow("mergeright", merge);
                cv::line(merge, cvPoint(om1.x, om1.y), cvPoint(mm.x + tmp.cols, mm.y), cv::Scalar(255, 255, 0), 1);
                cv::line(merge, cvPoint(mm.x, mm.y), cvPoint(om2.x + tmp.cols, om2.y), cv::Scalar(127, 255, 0), 1);

                cv::putText(merge, sss, cv::Point(20, 20), font_face, 1, cv::Scalar(0, 255, 255), 2);

                imshow("merge", merge);
                imwrite("merge.png", merge);

                //2D图像旋转变换
                mvTransform2DImage((unsigned char*)tmp.data, tmp.cols, tmp.rows,
                    tmp.channels(), eval->hmat);
                imshow("org-img-tran", input2);
                imwrite("org-img-tran.png", input2);

                mvImage cc;
                mvPoint ee;
                cc.height = input2.rows;
                cc.width = input2.cols;
                cc.channels = input2.channels();
                cc.pdata = input2.data;
                cc.nsize = input2.step * input2.rows;
                ee.x = input2.cols / 2; ee.y = input2.rows / 2;
                //任意2D图象变换，cc位旋转轴点,angle = -45, scale = 0.8
                mvRotate2DImage(cc, ee, -45, 0.3);
                imwrite("2d-trans.png", input2);

            }
            cv::waitKey();
        }
    }

    key = cv::waitKey(0);

    while (key != 'q' && key != 'Q')
    {
        cv::waitKey(0);
    }

    return 0;
}
