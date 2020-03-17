/************************************************************************
* File      :  demo.cpp
*
* 旋转和缺陷定位检测demo
*
* By Wsn
* 2017-OCT-27
************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <opencv2/opencv.hpp>
#include "../../3rdparty/DllMvInterface.h"
//#include "cvInterface.h"
#include <iostream>
#include <vector>
#include<opencv2/ml/ml.hpp>
#include <fstream>
#include <opencv2/imgproc/imgproc_c.h>


using namespace cv;
using namespace std;

static RNG rng(12345);

algDllHandle *pAlg = NULL;


//alg算法配置
//算法缩放系数
#define MV_ALG_SCL     1
//算法配件路径
//#define MV_CONFIG_PATH   "d:/imvcfg/"
//#define TEST_SAMPLE_PATH "D:/pic/"
//#define TEST_PATH "imageList_template.txt"

#define MV_CONFIG_PATH   "e:/AlgDemo/3rdparty/imvcfg/"
#define TEST_SAMPLE_PATH "e:/AlgDemo/3rdparty/pic/"
#define TEST_PATH "imageListtran.txt"

//输入图像缩放系数
#define MV_INTPUT_IMAGE_SCAL   1
//算法类型
#define MV_ALG_TYPE  (MV_ALG_DEFECTSDET)
#define INPUT_WIDTH  1024
#define INPUT_HEIGHT 768


typedef struct
{
    mvRGBImage tmpImg;       //template img
    mvRGBImage curImg;       //current process img
    float hmat[9];           //3x3 transorm mat
    float angle;             //rotate angle
    int reserved1;
    float reservd2;
}defectsUserResult;

/***
load images from file-list

*/
void load_images(const string & prefix, const string & filename, vector< Mat > & img_lst, int flag, Size ct, int *ww, int *hh)
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
        *ww = img.cols;
        *hh = img.rows;
    }
}


/***
draw mvCCLItems

*/
void cclItemsDraw(mvCCLItems *pComps, cv::Mat img)
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


    for (i = pComps->max_comp - 1; i >= 0; i--)
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
        //mvImageDrawRectangle(pImage, 255, rc.pntUpLft.y, rc.pntDnRgt.y,
        //  rc.pntUpLft.x, rc.pntDnRgt.x);
        cv::rectangle(img, cv::Rect(x, y, xx - x, yy - y), Scalar(0, 0, 255));
        for (int k = 0; k < pComp->pnt_countour.nump; k++)
        {
            //mvLog("maxOds:%f,", pods[k]);
            pp1.x = pComp->pnt_countour.pnts[k].x;
            pp1.y = pComp->pnt_countour.pnts[k].y;

            cv::line(img, pp1, pp1, color, 1, 8, 0);
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
        line(img, pp1, pp1, cv::Scalar(0, 0, 255), 2);
    }

    imshow("ccls", img);

    return;
}


void* algProcessDemo(algType type)
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
    int ww, hh;
    defectsUserResult *eval;
    std::string text;
    char sss[100];

    string test_dir = TEST_SAMPLE_PATH;
    string test_sample = TEST_PATH;

    //ret = FDLLMvLibLoad();
    //if (ret < 1)
    //  return 0;

    //设置alg param初始化参数
    strcpy_s(param.cfg_path, MV_CONFIG_PATH);
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
    if (0)
    {
        //读取本地图像
        load_images(test_dir, test_sample, test_lst, 1, Size(INPUT_WIDTH, INPUT_HEIGHT), &ww, &hh);

        pAlg = (algDllHandle*)mvInstanceAlloc(INPUT_WIDTH, INPUT_HEIGHT, type, &param);
        if (!pAlg)
            return NULL;

        pAlg->alg_params.disp_level |= (0x01 << 21);
        //pAlg->algParam.disLevel |= (0x01 << 22);
        pAlg->alg_params.disp_level &= ~(0x01 << 22);
        pRes = (mvResult*)&pAlg->result;
        /* 算法结果输出 */
        eval = (defectsUserResult*)pRes->user_dat;
    }
    else
    {

        load_images(test_dir, test_sample, test_lst, 0, Size(INPUT_WIDTH, INPUT_HEIGHT), &ww, &hh);

        pAlg = (algDllHandle*)mvInstanceAlloc(ww, hh, type, &param);
        if (!pAlg)
            return NULL;

        pAlg->alg_params.disp_level |= (0x01 << 21);
        pAlg->alg_params.disp_level &= ~(0x01 << 22);

        pRes = (mvResult*)&pAlg->result;
        /* 算法结果输出 */
        eval = (defectsUserResult*)pRes->user_dat;
    }

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


        orgImage.index = frameNum;           /* index = 0,  template, */
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

        if (ret > 0)
        {//MV_OK
            cv::Mat tmp;
            if (eval->tmpImg.channels == 1)
                tmp = Mat(eval->tmpImg.height, eval->tmpImg.width, CV_8UC1, (void*)eval->tmpImg.pdata);
            else if (eval->tmpImg.channels == 3)
                tmp = Mat(eval->tmpImg.height, eval->tmpImg.width, CV_8UC3, (void*)eval->tmpImg.pdata);
            imshow("tmp-img", tmp);

            /* 算法处理结果输出 */
            eval = (defectsUserResult*)pRes->user_dat;
            sprintf_s(sss, "agnle = %.4f", eval->angle);
            cout << sss << endl;

            //绘制defects检测结果
            //mvMatchObjsDrawAndDisplay(pAlg, pRes);

            //绘制组件
            //cclTtemsDraw(pRes->ccls, input2);
            imwrite("ccls.png", input2);
            //显示defects检测结果
            imshow("det-result", frame);
            imwrite("det-result.png", input2);

            if (frameNum)
            {
                int font_face = cv::FONT_HERSHEY_SIMPLEX;
                mvFPoint mm, om1, om2;

                //imshow("org-img", input2);
                //标记点选择
                mm.x = input2.cols / 2 / 2;  mm.y = input2.rows / 2 / 2;
                cv::line(input2, cvPoint(mm.x, mm.y), cvPoint(mm.x, mm.y), cv::Scalar(0, 255, 0), 2);
                //cv::circle(input2, cvPoint(mm.x, mm.y), 3, cv::Scalar(0, 255, 0));
                cv::line(tmp, cvPoint(mm.x, mm.y), cvPoint(mm.x, mm.y), cv::Scalar(0, 255, 255), 2);
                //cv::circle(tmp, cvPoint(mm.x, mm.y), 2, cv::Scalar(0, 255, 255));
                cv::imwrite("mark1.bmp", input2);
                cv::imwrite("mark2.bmp", tmp);
                //计算模板图中坐标om1
                mvMap2DPoint(mm, (double *)eval->hmat, 0, om1);
                cv::line(tmp, cvPoint(om1.x, om1.y), cvPoint(om1.x, om1.y), cv::Scalar(0, 255, 0), 2);
                //cv::circle(tmp, cvPoint(om1.x, om1.y), 3, cv::Scalar(0, 255, 0));
                cv::imwrite("mark3.bmp", tmp);
                //计算检测图中坐标om2
                mvMap2DPoint(mm, (double*)eval->hmat, 1, om2);
                cv::line(input2, cvPoint(om2.x, om2.y), cvPoint(om2.x, om2.y), cv::Scalar(0, 255, 255), 2);
                //cv::circle(input2, cvPoint(om2.x, om2.y), 3, cv::Scalar(0, 255, 255));
                cv::imwrite("mark4.bmp", input2);
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
                mvTransform2DImage((unsigned char*)input2.data, input2.cols, input2.rows,
                    input2.channels(), (double *)eval->hmat);
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


            }

            frameNum++;
            //cvWaitKey();

        }
    }

    if (1)
    {
        printf("press anykey to continue...\n");
        cv::waitKey();
        //FDLLMvInstanceDelete(pAlg);
        //FDLLMvLibFree();
    }
    printf("succussfully destroy alg handle!\n");
    while (1);
    return (void*)pAlg;
}

int main(int argc, char* argv[])
{
    int key;


    ////process
    algProcessDemo(MV_ALG_TYPE);

    key = cv::waitKey(0);

    while (key != 'q' && key != 'Q')
    {
        cv::waitKey(0);
    }
    return 0;
}
