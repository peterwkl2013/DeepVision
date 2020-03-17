﻿/************************************************************************
* File      : alg demo .c
*
*
* By WSN
* 2019-May-19
************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <opencv2/core.hpp>


#include <opencv2/core/types_c.h>
#include <opencv2/core/core_c.h>

#include <opencv2/opencv.hpp>
#include <windows.h>
#include "mvSDK_interface.h"

#include <fstream>
#include <iostream>
#include <vector>
#include<opencv2/ml/ml.hpp>

using namespace cv;
using namespace std;


RNG rng(12345);


/* 算法配件路径 */

#define MV_CONFIG_PATH   R"(E:\AlgDemo\3rdparty\imvcfg\)"
#define TEST_SAMPLE_PATH R"(E:\AlgDemo\3rdparty\pic\ccl\)"


#define TEST_PATH "imageList.txt"

#define INPUT_WIDTH 1280
#define INPUT_HEIGHT 960

extern void mvComponentsProcess2(Mat& img, vector<mvCCLObject>& objs, int edge_thres, int k1, int k2);

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

void algLocationProcessWrap(algType type)
{
    int ret;
    mvInputImage orgImage;
    //algDllHandle *pAlg = NULL;
    initParam param;
    mvResult *pRes;
    mvEngineCfg  pa;
    int keyval = 0;
    int key;
    IplImage *tmporg;
    int width = 80;
    int height = 96;
    CvCapture *pCap;
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


    /* 读取本地图像 */
    load_images(test_dir, test_sample, test_lst, 0, Size(INPUT_WIDTH, INPUT_HEIGHT));

    vector< Mat >::const_iterator img = test_lst.begin();
    vector< Mat >::const_iterator end = test_lst.end();
    vector<mvCCLObject> objs;
    frameNum = 0;
    mvDetRoi det;

    for (int i = 0; img != end; img++)
    {
        static int flag = 0;
        cv::Mat frame = img->clone();
        cv::Mat frame_copy = img->clone();
        int64 nTick = ::GetTickCount();

        double ptime;
        nTick = getTickCount();
        //mvComponentsProcess(frame, objs, 100, 0, 0, 0);   /* 提取特征 */

        mvComponentsProcess2(frame, objs, 100);
        ptime = ((double)getTickCount() - nTick)*1000. / getTickFrequency();
        printf("processed time = %.2f ms\n", ptime);

        for (int ii = 0; ii < objs.size(); ii++)
        {
            mvCCLObject obj = objs[ii];
            vector<Point> cc = obj.contours;

            for (int kk = 0; kk < cc.size(); kk++)
            {
                circle(frame, cc[kk], 1, Scalar(0, 255, 0));
            }

            //cout << "wh_rate:" << obj.wh_reate << " roundness:" << obj.roundness << endl;

            /* 目标过滤条件 */
            //if (obj.wh_reate * obj.roundness > 0.9)
            //{
            //  for (int j = 0; j < 4; j++)
            //      line(frame, obj.box_pnts[j], obj.box_pnts[(j + 1) % 4], Scalar(0, 0, 255), 2, 8);
            //}
            //else
            {
                for (int j = 0; j < 4; j++)
                    line(frame, obj.box_pnts[j], obj.box_pnts[(j + 1) % 4], Scalar(255, 0, 0), 2, 8);
            }
        }
        imshow("components", frame);
        waitKey();
    }

    key = waitKey();


    printf("stop.\n");
    while (1);


    return;
}

int main(int argc, char* argv[])
{
    int key;


    ////process
    algLocationProcessWrap((algType)-1);

    key = waitKey(0);

    while (key != 'q' && key != 'Q')
    {
        waitKey(0);
    }

    return 0;
}
