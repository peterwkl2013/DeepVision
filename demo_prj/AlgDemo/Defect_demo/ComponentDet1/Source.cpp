/************************************************************************
* File      : alg demo .c
*
*
* By Wsn
* 2017-Jul-11
************************************************************************/

#include <cstdio>
#include <cstdlib>
#include <opencv2/opencv.hpp>
#include "../../3rdparty/DllMvInterface.h"
#include <iostream>
#include <vector>
#include<opencv2/ml/ml.hpp>
#include <fstream>
#include <opencv2/imgproc/imgproc_c.h>



using namespace cv;
using namespace std;

algDllHandle *pAlg = NULL;
RNG rng(12345);

//alg算法配置
//算法缩放系数
#define MV_ALG_SCL     1

//算法配件路径

#define MV_CONFIG_PATH "E:/AlgDemo/3rdparty/imvcfg/"
#define TEST_SAMPLE_PATH "E:/AlgDemo/3rdparty/pic/"
//输入图像缩放系数
#define MV_INTPUT_IMAGE_SCAL   1


//算法类型
#define MV_ALG_TYPE  (MV_ALG_COMPONENT_DET2)
#define INPUT_WIDTH  2448
#define INPUT_HEIGHT 2048

//
////
//#define MV_CONFIG_PATH   "d:/imvcfg/"
//#define TEST_SAMPLE_PATH "D:/pic/xzb/"


#define TEST_PATH "imageListComponent.txt"



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
            //imshow("image", img2);
            //waitKey(10);
#endif
            img_lst.push_back(img2.clone());
        }
        else
            img_lst.push_back(img.clone());
    }
}

void cclTtemsDraw(mvCCLItems *pComps, cv::Mat img)
{
    int i;
    mvCCLItem *pComp;
    mvRect rc;
    mvPoint pt;
    int x, xx, y, yy;
    cv::Point pp1, pp2;
    vector<mvFPoint> subpixels;

    //mvSubpixelsExtract(pComps->pnts_bank, pComps->width, pComps->height, subpixels);

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
            //cv::line(img, pp1, pp1, color, 1, 8, 0);
            cv::circle(img, pp1, 0.1, color, 1, 8, 0);
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

    return;
}

void* algLocationProcessWrap(algType type)
{
    int ret;
    mvInputImage orgImage;
    //algDllHandle *pAlg = NULL;
    initParam param;
    mvResult *pRes;
    int keyval = 0;
    int key;
    int width = 80;
    int height = 96;

    mvImage clone;

    vector< Mat > pos_lst;
    vector< Mat > test_lst;
    vector< Mat > full_neg_lst;
    vector< Mat > neg_lst;
    vector< Mat > gradient_lst;
    vector< int > labels;

    //string neg = NEG_PATH;
    string test_dir = TEST_SAMPLE_PATH;
    string test_sample = TEST_PATH;
    int frameNum;


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
    param.tmp_img.type = MV_CV_IPL;        //MV_CV_IPL;

                                           //allocation 512X512
    IplImage *org = NULL;
    pAlg = (algDllHandle*)mvInstanceAlloc(INPUT_WIDTH, INPUT_HEIGHT, type, &param);
    if (!pAlg)
        return NULL;

    pAlg->sub_type = 2;   //sub_type of this algorithm

    pRes = (mvResult*)&pAlg->result;


    //读取本地图像
    load_images(test_dir, test_sample, test_lst, 1, Size(INPUT_WIDTH, INPUT_HEIGHT));

    vector< Mat >::const_iterator img = test_lst.begin();
    vector< Mat >::const_iterator end = test_lst.end();
    frameNum = 0;
    mvDetRoi det;

    for (int i = 0; img != end; img++)
    {
        static int flag = 0;
        //cv::Mat frame = cv::imread(TEST_IMAGE, -1);
        cv::Mat frame = img->clone();
        if (flag == 0)
        {
            org = cvCreateImage(cvSize(frame.cols, frame.rows), IPL_DEPTH_8U, 3);
            flag = 1;


            det.num_poly = 3;
            int startxy = 120;
            int rww, rhh;
            rww = 200; rhh = 200;

            det.polys[0].ppnts[0].x = 40;
            det.polys[0].ppnts[0].y = 40;

            det.polys[0].ppnts[1].x = rww - det.polys[0].ppnts[0].x;
            det.polys[0].ppnts[1].y = det.polys[0].ppnts[0].y;
            det.polys[0].ppnts[2].x = rww - det.polys[0].ppnts[0].x;
            det.polys[0].ppnts[2].y = rhh - det.polys[0].ppnts[0].y;
            det.polys[0].ppnts[3].x = det.polys[0].ppnts[0].x;
            det.polys[0].ppnts[3].y = rhh - det.polys[0].ppnts[0].y;
            det.polys[0].ppnts[4] = det.polys[0].ppnts[0];
            det.polys[0].num = 5;
            det.polys[0].index = 0;
            det.polys[0].uc = 80;
            det.polys[0].valid = 1;
            det.polys[0].uflag = 0;
            det.polys[0].seed.x = det.polys[0].ppnts[0].x + 2;
            det.polys[0].seed.y = det.polys[0].ppnts[0].y + 2;

            cv::Mat frame2 = img->clone();

            clone.channels = 1;
            clone.width = frame2.cols;
            clone.height = frame.rows;

            det.roi_map = NULL; //important

                                //det.roi_map = NULL; //important
                                //mvSetDetRoiArea(pAlg, det, NULL);

        }

        Mat input2;

        frame.copyTo(input2);

        orgImage.index = frameNum++;
        orgImage.pframe = (void*)frame.data;        /* 图像数据地址 */
        orgImage.width = frame.cols;                /* 图像宽度 */
        orgImage.height = frame.rows;               /* 图像高度 */
        orgImage.channels = frame.channels();       /* 图像通道*/
        orgImage.wstep = frame.cols * frame.channels();  /* 图像 wstep = 宽度* 通道数 */

        orgImage.depth = frame.depth();         /* 图像深度 */
        orgImage.type = MV_BGR24;   /*帧的格式*/

        int64 nTick = cv::getTickCount();

        double ptime;
        nTick = getTickCount();

        /* 算法处理 */
        imshow("org", frame);
        ret = mvAlgProcess(pAlg, (mvInputImage*)&orgImage);
        ptime = ((double)getTickCount() - nTick)*1000. / getTickFrequency();
        printf("processed time = %.2f ms\n", ptime);
        cout << pAlg->result.cc_items->ccomps->com_label;
        if (ret > 0)
        {
            //绘制结果
            //mvMatchObjsDrawAndDisplay(pAlg, pRes);

            cclTtemsDraw(pRes->cc_items, input2);
            imshow("ccls", input2);

            //strcpy(strsss, TEST_SAMPLE_PATH);
            //sprintf(nnn, "result-%02d.png", frameNum);
            //strcat(strsss, nnn);
            //
            //imwrite(strsss, input2);

            Mat bin, edge, com;

            bin = Mat(cvSize(pAlg->bin_img.width, pAlg->bin_img.height), CV_8UC1, (void*)pAlg->bin_img.pdata);
            imshow("bin-image", bin);
            //strcpy(strsss, TEST_SAMPLE_PATH);
            //sprintf(nnn, "bin-%02d.png", frameNum);
            //strcat(strsss, nnn);
            //imwrite(strsss, bin);

            edge = Mat(cvSize(pAlg->edge_img.width, pAlg->edge_img.height), CV_8UC1, (void*)pAlg->edge_img.pdata);
            imshow("edge-image", edge);
            //com = Mat(cvSize(pAlg->com_img.width, pAlg->com_img.height), CV_8UC1, (void*)pAlg->com_img.pdata);
            //threshold(com, com, 0, 255, THRESH_BINARY);
            //imshow("com-image", com);

            //org->imageData = tmpptr;

            mvImage imm;

            mvImageCreate(&imm, pAlg->com_img.width, pAlg->com_img.height, 1);
            mvSetComponentRoiArea(imm, det);
            mvImageShow("mvImage", imm);
            mvImageDestroy(&imm);

            /*Mat src = frame.clone();
            cvtColor(src, src, CV_BGR2GRAY);
            mvImage testImg;
            mvImageCreate(&testImg, src.cols, src.rows, src.channels());
            mvSetComponentRoiArea(testImg, det);
            for (auto i = 0; i < testImg.height; i++)
            {
            for (auto j = 0; j < testImg.width; j++)
            {

            }
            }
            mvImageShow("testImg", testImg);
            mvImageDestroy(&testImg);*/

            int cnt = mvDetRoiMapFilter(det, pAlg->com_img.width, pAlg->com_img.height, pAlg->com_img.pdata);
            cout << "cnt=" << cnt << endl;
            com = Mat(cvSize(pAlg->com_img.width, pAlg->com_img.height), CV_8UC1, (void*)pAlg->com_img.pdata);

            /* Mat srcTest = frame.clone();
            cvtColor(srcTest, srcTest, CV_BGR2GRAY);
            cnt = mvDetRoiMapFilter(det, srcTest.cols, srcTest.rows, srcTest.data);
            imshow("srctest", srcTest);*/
            //threshold(com, com, 0, 255, THRESH_BINARY);

            ///* 将ccl 的lable 二值化 */
            for (i = 0; i < pAlg->com_img.nsize; i++)
            {
                if (pAlg->com_img.pdata[i] > 0)
                    pAlg->com_img.pdata[i] = 255;
            }
            for (int i = 0; i < com.cols; i++)
            {
                for (int j = 0; j < com.rows; j++)
                {
                    com.at<uchar>(i, j);
                }
            }


            imshow("com-image", com);
            key = cv::waitKey();

        }
    }
    while (1);
    key = cv::waitKey();


    printf("stop.\n");
    cv::waitKey();

    if (1)
    {
        mvInstanceDelete(pAlg);
    }


    return (void*)pAlg;
}

int main(int argc, char* argv[])
{
    int key;


    ////process
    algLocationProcessWrap(MV_ALG_TYPE);

    key = cv::waitKey(0);

    while (key != 'q' && key != 'Q')
    {
        cv::waitKey(0);
    }

    return 0;
}
