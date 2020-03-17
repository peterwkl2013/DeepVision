/************************************************************************
* File      :  demo.cpp
*
* 旋转和缺陷定位检测demo
*
* By Wsn
* 2017-OCT-27
************************************************************************/
#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <opencv2/opencv.hpp>
#include "../../3rdparty/DllMvInterface.h"
#include <iostream>
#include <fstream>
#include <vector>
#include<opencv2/ml/ml.hpp>
#include <thread>
#include <opencv2/imgproc/imgproc_c.h>



using namespace cv;
using namespace std;

algDllHandle *pAlg = NULL;


//alg算法配置
//算法缩放系数
#define MV_ALG_SCL     1
//算法配件路径
#define MV_CONFIG_PATH   "E:/AlgDemo/3rdparty/imvcfg/"
//输入图像缩放系数
#define MV_INTPUT_IMAGE_SCAL   1
//算法类型
#define MV_ALG_TYPE  (MV_ALG_DEFECTSDET)
#define INPUT_WIDTH  1024
#define INPUT_HEIGHT 750
#define TEST_SAMPLE_PATH "E:/AlgDemo/3rdparty/pic/"
#define TEST_PATH "imageList_template.txt"

typedef struct
{
    mvRGBImage tmp_img;       //template img
    mvRGBImage curImg;       //current process img
    float hmat[9];           //3x3 transorm mat
    float angle;             //rotate angle
    int reserved1;
    float reservd2;
}defectsUserResult;

void load_images(const string & prefix, const string & filename, vector< Mat > & img_lst, int flag, Size ct)
{
    string line;
    ifstream file;
    Mat img1, img2;
    int width = ct.width;
    int height = ct.height;


    file.open((prefix + filename).c_str());
    if (!file.is_open())
    {
        cerr << "Unable to open the list of images from " << filename << " filename." << endl;
        exit(-1);
    }

    bool end_of_parsing = false;
    while (!end_of_parsing)
    {
        getline(file, line);
        if (line.empty()) // no more file to read
        {
            end_of_parsing = true;
            break;
        }
        Mat img = imread((prefix + line).c_str()); // load the image
        if (img.empty()) // invalid image, just skip it.
            continue;
#ifndef _DEBUGX
        //imshow("image", img);
        //waitKey(10);
#endif
        if (flag)
        {
            //resize
            resize(img, img2, Size(width, height));
#ifdef _DEBUG
            imshow("image", img2);
            waitKey(10);
#endif
            img_lst.push_back(img2.clone());
        }
        else
            img_lst.push_back(img.clone());
    }
}


void SetROI()
{
    mvDetRoi roi;
    roi.num_poly = 1;
    roi.roi_map = NULL;

    int width = 200;
    int height = 200;

    //width *= m_alg_scale;
    //height *= m_alg_scale;

    roi.polys[0].index = 0;
    roi.polys[0].uc = 50;
    roi.polys[0].valid = 1;
    roi.polys[0].uflag = 0;
    roi.polys[0].seed.x = 700;
    roi.polys[0].seed.y = 500;

    roi.polys[0].num = 5;
    roi.polys[0].ppnts[0].x = 600;
    roi.polys[0].ppnts[0].y = 400;
    roi.polys[0].ppnts[1].x = 600 + width;
    roi.polys[0].ppnts[1].y = 400;
    roi.polys[0].ppnts[2].x = 600 + width;
    roi.polys[0].ppnts[2].y = 400 + height;
    roi.polys[0].ppnts[3].x = 600;
    roi.polys[0].ppnts[3].y = 400 + height;
    roi.polys[0].ppnts[4] = roi.polys[0].ppnts[0];
    mvSetDetRoiArea(pAlg, roi, MV_CONFIG_PATH);
}

void DrawDefectResult(cv::Mat & img)
{
    if (!pAlg || !img.data)
    {
        return;
    }
    mvResult* pRes = (mvResult*)&pAlg->result;
    defectsUserResult* eval = (defectsUserResult*)pRes->user_dat;
    int lineLength = 10;
    if (pRes->mat_objs.num_valid > 0)
    {
        for (auto i = 0; i < pRes->mat_objs.num_obj; i++)
        {
            matchObj *moj = &pRes->mat_objs.mat_obj[i];
            vector<cv::Point> pnts;
            mvFPoint inPnt, outPnt;
            for (int i = 0; i < 4; i++)
            {
                inPnt.x = moj->rot_box.pnts[i].x;
                inPnt.y = moj->rot_box.pnts[i].y;
                pnts.push_back(cv::Point(inPnt.x, inPnt.y));
            }
            inPnt.x = moj->center.x;
            inPnt.y = moj->center.y;
            cv::Point centerPoint = cv::Point(inPnt.x, inPnt.y);

            cv::polylines(img, pnts, true, cv::Scalar(0, 0, 255), 2);
            cv::line(img, centerPoint, cv::Point(centerPoint.x, centerPoint.y - lineLength), cv::Scalar(0, 255, 0), 2);
            cv::line(img, centerPoint, cv::Point(centerPoint.x + lineLength, centerPoint.y), cv::Scalar(0, 255, 0), 2);
            cv::line(img, centerPoint, cv::Point(centerPoint.x, centerPoint.y + lineLength), cv::Scalar(0, 255, 0), 2);
            cv::line(img, centerPoint, cv::Point(centerPoint.x - lineLength, centerPoint.y), cv::Scalar(0, 255, 0), 2);
            cv::line(img, centerPoint, centerPoint, cv::Scalar(0, 0, 255), 3);

        }
    }
    mvTransform2DImage((unsigned char*)img.data, img.cols, img.rows, img.channels(), (double*)&eval->hmat);
}



void* algLocationProcessWrap(algType type)
{
    int ret;
    mvInputImage orgImage;
    initParam param;
    mvResult *pRes;
    mvEngineCfg  pa;
    int keyval = 0, key;
    CvCapture *pCap;
    vector< Mat > test_lst;
    int frameNum;

    string test_dir = TEST_SAMPLE_PATH;
    string test_sample = TEST_PATH;

    /* ret = FDLLMvLibLoad();
    if (ret < 1)
    return 0;*/

    //设置alg param初始化参数
    strcpy(param.cfg_path, MV_CONFIG_PATH);
    //param.tmpOffset        = offset;
    //param.tmpRoi           = loc;
    param.online_temp = 0;    //not used
    param.tmp_img.pframe = NULL;
    param.tmp_img.width = 0;        //temp->width;
    param.tmp_img.height = 0;       //temp->height;
    param.tmp_img.channels = 0;     //temp->nChannels;
    param.tmp_img.depth = 0;        //temp->depth;
    param.tmp_img.wstep = 0;    //temp->widthStep;
    param.tmp_img.type = MV_CV_IPL;        //MV_CV_IPL;

                                           //allocation INPUT_WIDTH X INPUT_HEIGHT
    IplImage *org = NULL;
    if (pAlg == NULL)
    {
        pAlg = (algDllHandle*)mvInstanceAlloc(INPUT_WIDTH, INPUT_HEIGHT, MV_ALG_DEFECTSDET, &param);
    }
    if (!pAlg)
        return NULL;

    //SetROI();

    pAlg->alg_params.disp_level |= (0x01 << 21);
    //pAlg->alg_params.disp_level |= (0x01 << 22);
    pAlg->alg_params.disp_level &= ~(0x01 << 22);

    defectsUserResult *eval;
    pRes = (mvResult*)&pAlg->result;
    /* 算法结果输出 */
    eval = (defectsUserResult*)pRes->user_dat;

    //读取本地图像
    load_images(test_dir, test_sample, test_lst, 1, Size(INPUT_WIDTH, INPUT_HEIGHT));

    vector< Mat >::const_iterator img = test_lst.begin();
    vector< Mat >::const_iterator end = test_lst.end();
    frameNum = 0;
    for (int i = 0; img != end; img++)
    {
        static int flag = 0;
        //cv::Mat frame = cv::imread(TEST_IMAGE, -1);
        cv::Mat frame = img->clone();
        cv::Mat input2 = img->clone();
        if (flag == 0)
        {
            org = cvCreateImage(cvSize(frame.cols, frame.rows), IPL_DEPTH_8U, 3);
            flag = 1;

        }
        orgImage.index = frameNum++;           /* index = 0,  template, */
        orgImage.pframe = (void*)frame.data;        /* 图像数据地址 */
        orgImage.width = frame.cols;                /* 图像宽度 */
        orgImage.height = frame.rows;               /* 图像高度 */
        orgImage.channels = frame.channels();       /* 图像通道*/
        orgImage.wstep = frame.cols * frame.channels();  /* 图像 widthStep = 宽度* 通道数 */

        orgImage.depth = frame.depth();         /* 图像深度 */
        orgImage.type = MV_BGR24;   /*帧的格式*/

        int64 nTick = cv::getTickCount();

        double ptime;
        nTick = getTickCount();

        /* 算法处理 */
        ret = mvAlgProcess(pAlg, (mvInputImage*)&orgImage);
        ptime = ((double)getTickCount() - nTick)*1000. / getTickFrequency();
        printf("processed time = %.2f ms\n", ptime);

        if (ret > 0 && frameNum > 1)
        {
            /* 算法处理结果输出 */
            eval = (defectsUserResult*)pRes->user_dat;
            printf("agnle = %.4f\n", eval->angle);
            //绘制defects检测结果
            //mvMatchObjsDrawAndDisplay(pAlg, pRes);

            DrawDefectResult(frame);


            //显示defects检测结果
            imshow("result", frame);
            //2D图像旋转变换
            imshow("org-img", input2);       //旋转待检测图像
            mvTransform2DImage((unsigned char*)input2.data, input2.cols, input2.rows,
                input2.channels(), (double*)&eval->hmat);
            imshow("org-img-tran", input2);   //旋转后图像

            cv::Mat tmp;
            if (eval->tmp_img.channels == 1)
                tmp = Mat(eval->tmp_img.height, eval->tmp_img.width, CV_8UC1, (void*)eval->tmp_img.pdata);
            else if (eval->tmp_img.channels == 3)
                tmp = Mat(eval->tmp_img.height, eval->tmp_img.width, CV_8UC3, (void*)eval->tmp_img.pdata);
            imshow("tmp-img", tmp);
            cv::waitKey();

        }
    }
    return (void*)pAlg;
}

int main(int argc, char* argv[])
{
    int key;

    algLocationProcessWrap(MV_ALG_TYPE);

    key = cv::waitKey(0);

    while (key != 'q' && key != 'Q')
    {
        cv::waitKey(0);
    }

    return 0;
}
