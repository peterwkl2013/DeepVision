/************************************************************************
* File      : alg demo .c
*
*
* By Wsn
* 2017-Jul-11
************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <opencv2/core.hpp>


#include <opencv2/core/types_c.h>
#include <opencv2/core/core_c.h>

#include <opencv2/opencv.hpp>
#include "DllmvInterface.h"
#include <windows.h>
#include "cvInterface.h"
#include "mvSDK_interface.h"

#include <fstream>
#include <iostream>
#include <vector>
#include<opencv2/ml/ml.hpp>

using namespace cv;
using namespace std;

algDllHandle *pAlg = NULL;
RNG rng(12345);

//alg算法配置
//算法缩放系数
#define MV_ALG_SCL     1

//算法配件路径

#define MV_CONFIG_PATH   "./imvcfg/"
#define TEST_SAMPLE_PATH "./pic/"
//输入图像缩放系数
#define MV_INTPUT_IMAGE_SCAL   1


//算法类型
#define MV_ALG_TYPE  (MV_ALG_ZMCLOTH_SEGNET)
#define INPUT_WIDTH  473
#define INPUT_HEIGHT 473

//
////
#define MV_CONFIG_PATH   "./pic/imvcfg/"
//#define TEST_SAMPLE_PATH "D:/pic/xzb/"

#define TEST_SAMPLE_PATH "./pic/road/"


#define TEST_PATH "imageList.txt"


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
		//	rc.up_left.x, rc.dw_right.x);
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
		line(img, pp1, pp1, cvScalar(0, 0, 255), 2);
	}

	return;
}

std::vector<std::string> classes;
std::vector<Vec3b> colors;

void colorizeSegmentation(const Mat &score, Mat &segm)
{
	const int rows = score.size[2];
	const int cols = score.size[3];
	const int chns = score.size[1];

	if (colors.empty())
	{
		// Generate colors.
		colors.push_back(Vec3b());
		for (int i = 1; i < chns; ++i)
		{
			Vec3b color;
			for (int j = 0; j < 3; ++j)
				color[j] = (colors[i - 1][j] + rand() % 256) / 2;
			colors.push_back(color);
		}
	}
	else if (chns != (int)colors.size())
	{
		CV_Error(Error::StsError, format("Number of output classes does not match "
			"number of colors (%d != %zu)", chns, colors.size()));
	}

	Mat maxCl = Mat::zeros(rows, cols, CV_8UC1);
	Mat maxVal(rows, cols, CV_32FC1, score.data);
	for (int ch = 1; ch < chns; ch++)
	{
		for (int row = 0; row < rows; row++)
		{
			const float *ptrScore = score.ptr<float>(0, ch, row);
			uint8_t *ptrMaxCl = maxCl.ptr<uint8_t>(row);
			float *ptrMaxVal = maxVal.ptr<float>(row);
			for (int col = 0; col < cols; col++)
			{
				if (ptrScore[col] > ptrMaxVal[col])
				{
					ptrMaxVal[col] = ptrScore[col];
					ptrMaxCl[col] = (uchar)ch;
				}
			}
		}
	}

	segm.create(rows, cols, CV_8UC3);
	for (int row = 0; row < rows; row++)
	{
		const uchar *ptrMaxCl = maxCl.ptr<uchar>(row);
		Vec3b *ptrSegm = segm.ptr<Vec3b>(row);
		for (int col = 0; col < cols; col++)
		{
			ptrSegm[col] = colors[ptrMaxCl[col]];
		}
	}
}

void* algLocationProcessWrap(algType type)
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
	VideoCapture cap;
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
	strcpy(param.cfg_path, MV_CONFIG_PATH);
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

	ret = mvCheckXVisonLicense();
	if (ret == 0)
		cout << "no license-dog!" << endl;
	else if (ret == 2)
		cout << " license is not valid, please contact supportors" << endl;
	else if (ret == 1)
		cout << "license valid" << endl;

	pAlg = (algDllHandle*)mvInstanceAlloc(INPUT_WIDTH, INPUT_HEIGHT, type, &param);
	if (!pAlg)
		return NULL;

	pAlg->sub_type = 2;   //sub_type of this algorithm

	pRes = (mvResult*)&pAlg->result;
	frameNum = 0;
	mvDetRoi det;
	cv::Mat frame;
	Mat segm;
	int class_id = 21;     /* 支持21种物体识别分割 */

/* 类别着色 */
	if (colors.empty())
	{
		for (int label = 0; label < class_id; label++)
		{
			Vec3b color;
			int c;
			int j;
			color[0] = color[1] = color[2] = 0;
			c = label;
			j = 0;
			while (c)
			{
				color[0] |= ((c >> 0) & 1) << (7 - j);
				color[1] |= ((c >> 1) & 1) << (7 - j);
				color[2] |= ((c >> 2) & 1) << (7 - j);
				c >>= 3;
				j += 1;
			}
			colors.push_back(color);
		}
	}

#if 1
	//读取本地图像,并统一到相同尺寸进行处理
	load_images(test_dir, test_sample, test_lst, 0, Size(INPUT_WIDTH, INPUT_HEIGHT));

	vector< Mat >::const_iterator img = test_lst.begin();
	vector< Mat >::const_iterator end = test_lst.end();



	for (int i = 0; img != end; img++)
	{
		//{

			//cv::Mat frame = cv::imread(TEST_IMAGE, -1);
			frame = img->clone();
#else
	/* open camera */
	cap.open("test.avi");

	if (!cap.isOpened())
		cap.open(0);
	if (!cap.isOpened())
	{
		cout << "unble to open the camera." << endl;
		return NULL;
	}
	while (1)
	{
		cap >> frame;
#endif
		static int flag = 0;
		if (flag == 0)
		{
			org = cvCreateImage(cvSize(frame.cols, frame.rows), IPL_DEPTH_8U, 3);
			flag = 1;

			segm.create(frame.rows, frame.cols, CV_8UC3);
		}

		Mat input2;

		frame.copyTo(input2);

		//imshow("frame", frame);
		//waitKey();
		orgImage.index = frameNum++;
		orgImage.pframe = (void*)frame.data;        /* 图像数据地址 */
		orgImage.width = frame.cols;                /* 图像宽度 */
		orgImage.height = frame.rows;               /* 图像高度 */
		orgImage.channels = frame.channels();       /* 图像通道*/
		orgImage.wstep = frame.cols * frame.channels();  /* 图像 wstep = 宽度* 通道数 */

		orgImage.depth = frame.depth();         /* 图像深度 */
		orgImage.type = MV_BGR24;   /*帧的格式*/

		int64 nTick = ::GetTickCount();

		double ptime;
		nTick = getTickCount();

		/* 算法处理 */
		imshow("org", frame);

		// set alg sub-type
		pAlg->sub_type = 0;

		ret = mvAlgProcess(pAlg, (mvInputImage*)&orgImage);
		ptime = ((double)getTickCount() - nTick)*1000. / getTickFrequency();
		printf("processed time = %.2f ms\n", ptime);

		if (ret > 0)
		{
			char strsss[255];
			char nnn[255];
			Mat com;
			//绘制结果
			mvMatchObjsDrawAndDisplay(pAlg, pRes);

			////注意图象不是4的倍数的时候特殊处理！
			int widthStep;
			//com.create(pAlg->com_img.height, pAlg->com_img.width, CV_8UC(1));
			if (pAlg->com_img.width % 4 != 0)
			{
				widthStep = 4 - pAlg->com_img.width % 4;
				widthStep += pAlg->com_img.width;
			}
			else
				widthStep = pAlg->com_img.width;

			com = Mat(pAlg->com_img.height, pAlg->com_img.width, CV_8UC(1), (void*)pAlg->com_img.pdata, widthStep);
			int step = com.step;


			//for (int ii = 0; ii < pAlg->com_img.height; ii++)
			//{
			//	for (int jj = 0; jj < widthStep; jj++)
			//	{
			//		int index = ii * widthStep + jj;
			//		*com.ptr<uchar>(ii, jj) = pAlg->com_img.pdata[index];
			//	}
			//}
			Mat result_mat;

			//FILE *fp;
			//if ((fp = fopen("d:\\pic.dat", "wb")) != NULL)
			//{//保存原始图像数据
			//	fwrite(pAlg->com_img.pdata, pAlg->com_img.nsize,1, fp);
			//	fclose(fp);
			//}

			//FileStorage fs("d://vocabulary.xml", FileStorage::WRITE);
			//fs << "vocabulary" << com;
			//fs.release();


			result_mat.create(frame.rows, frame.cols, CV_8UC(1));
			resize(com, result_mat, result_mat.size(), 0, 0, INTER_NEAREST);   /*注意采用INTER_NEAREST方式) */

			//需要过滤掉result_mat中不是21种物体的标签(0~21)


			imshow("result_mat", result_mat);

			for (int row = 0; row < frame.rows; row++)
			{
				uchar *ptrMaxCl = result_mat.ptr<uchar>(row);
				Vec3b *ptrSegm = segm.ptr<Vec3b>(row);
				for (int col = 0; col < frame.cols; col++)
				{
					uchar val;
					if (ptrMaxCl[col])
						val = ptrMaxCl[col];
					Vec3b cc = colors[val];
					if (ptrMaxCl[col] < colors.size())
						ptrSegm[col] = colors[ptrMaxCl[col]];
				}
			}

			imshow("segm-resize", segm);

			addWeighted(frame, 0.40, segm, 0.8, 0.0, frame);
			//addWeighted(frame, 0.1, segm, 0.9, 0.0, frame);

			imshow("com-resize", result_mat);
			imshow("frame-resize", frame);
			key = waitKey(1);

			//mvInstanceDelete(pAlg);
		}
	}
	while (1);
	key = waitKey();


	printf("stop.\n");
	waitKey();
	
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

	key = waitKey(0);

	while (key != 'q' && key != 'Q')
	{
		waitKey(0);
	}

	return 0;
}

