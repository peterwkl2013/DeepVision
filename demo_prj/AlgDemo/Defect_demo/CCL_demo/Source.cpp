#include <opencv2/opencv.hpp>
#include "../../3rdparty/DllMvInterface.h"
#include <iostream>


using namespace cv;
using namespace std;

static RNG rng(12345);

algDllHandle *pAlg = NULL;
initParam param;
#define MV_CONFIG_PATH "E:/AlgDemo/3rdparty/imvcfg/"
#define MV_ALG_TYPE MV_ALG_COMPONENT_DET1
//#define MV_ALG_TYPE MV_ALG_DEFECTSDET
#define INPUT_WIDTH  560
#define INPUT_HEIGHT 524


typedef struct
{
    mvRGBImage tmpImg;       //template img
    mvRGBImage curImg;       //current process img
    float hmat[9];           //3x3 transorm mat
    float angle;             //rotate angle
    int reserved1;
    float reservd2;
}defectsUserResult;

void SetROI();

void InitParam()
{
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
}

int CCLDetect(const cv::Mat& img)
{
    pAlg = (algDllHandle*)mvInstanceAlloc(INPUT_WIDTH, INPUT_HEIGHT, MV_ALG_TYPE, &param);
    if (!pAlg)
        return NULL;
    SetROI();
    pAlg->alg_params.disp_level |= (0x01 << 21);
    pAlg->alg_params.disp_level &= ~(0x01 << 22);
    cv::Mat frame = img.clone();
    mvInputImage orgImage;
    orgImage.index = 0;           /* index = 0,  template, */
    orgImage.pframe = (void*)frame.data;        /* 图像数据地址 */
    orgImage.width = frame.cols;                /* 图像宽度 */
    orgImage.height = frame.rows;               /* 图像高度 */
    orgImage.channels = frame.channels();       /* 图像通道*/
    orgImage.wstep = frame.cols * frame.channels();  /* 图像 widthStep = 宽度* 通道数 */

    orgImage.depth = frame.depth();         /* 图像深度 */
    orgImage.type = MV_BGR24;   /*帧的格式*/
    int ret = mvAlgProcess(pAlg, (mvInputImage*)&orgImage);
    return ret;
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

void DrawResult(algDllHandle *pAlg, cv::Mat & img)
{
    if (!pAlg)
    {
        return;
    }
    mvResult* pRes = (mvResult*)&pAlg->result;
    int lineLenght = 10;
    cv::Point pp1, pp2;
    if (pRes->mat_objs.num_valid > 0)
    {
        cv::Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
        for (auto i = 0; i < pRes->mat_objs.num_obj; i++)
        {
            matchObj *moj = &pRes->mat_objs.mat_obj[i];
            /* cv::Rect rct;
            rct.x = moj->rec_bound.up_left.x;
            rct.y = moj->rec_bound.up_left.y;
            rct.width = moj->rec_bound.dw_right.x - moj->rec_bound.up_left.x;
            rct.height = moj->rec_bound.dw_right.y - moj->rec_bound.up_left.y;
            cv::rectangle(img, rct, color);*/
            for (int j = 0; j < moj->contours.nump; j++)
            {
                cv::line(img, cv::Point(moj->contours.pnts[j].x, moj->contours.pnts[j].y), cv::Point(moj->contours.pnts[j].x, moj->contours.pnts[j].y), color);

            }


            /*vector<cv::Point> pnts;
            for (int i = 0; i < 4; i++)
            {
            pnts.push_back(cv::Point(moj->rot_box.pnts[i].x, moj->rot_box.pnts[i].y));
            }
            cv::polylines(img, pnts, true, color, 1);*/



            //int len = 10;
            //pp1.x = moj->center.x;
            //pp1.y = moj->center.y - len;
            //pp2.x = moj->center.x;
            //pp2.y = moj->center.y + len;
            //rectangle(img, pp1, pp2, CV_RGB(0, 255, 0));
            //pp1.x = moj->center.x - len;
            //pp1.y = moj->center.y;
            //pp2.x = moj->center.x + len;
            //pp2.y = moj->center.y;
            //rectangle(img, pp1, pp2, CV_RGB(0, 255, 0));
            ///*object centor */
            //pp1.x = moj->center.x;
            //pp1.y = moj->center.y;
            //line(img, pp1, pp1, cvScalar(0, 0, 255), 2);


        }
    }
}

void SetROI()
{
    mvDetRoi roi;
    roi.num_poly = 1;
    roi.roi_map = NULL;

    int width = 560;
    int height = 524;

    roi.polys[0].index = 0;
    roi.polys[0].uc = 50;
    roi.polys[0].valid = 1;
    roi.polys[0].uflag = 0;
    roi.polys[0].seed.x = width / 2;
    roi.polys[0].seed.y = height / 2;

    roi.polys[0].num = 5;
    roi.polys[0].ppnts[0].x = width / 2 - 100;
    roi.polys[0].ppnts[0].y = height / 2 - 100;
    roi.polys[0].ppnts[1].x = width / 2 + 100;
    roi.polys[0].ppnts[1].y = height / 2 - 100;
    roi.polys[0].ppnts[2].x = width / 2 + 100;
    roi.polys[0].ppnts[2].y = height / 2 + 100;
    roi.polys[0].ppnts[3].x = width / 2 - 100;
    roi.polys[0].ppnts[3].y = height / 2 + 100;
    roi.polys[0].ppnts[4] = roi.polys[0].ppnts[0];

    mvSetDetRoiArea(pAlg, roi, MV_CONFIG_PATH);
}

int main(int argc, char* argv[])
{
    int key;

    InitParam();
    std::string path = "E:/AlgDemo/3rdparty/pic/bd1.bmp";
    cv::Mat srcImg = cv::imread(path.c_str(), IMREAD_GRAYSCALE);
    if (!srcImg.data)
    {
        return false;
    }
    cv::cvtColor(srcImg, srcImg, COLOR_GRAY2BGR);
    cv::Mat testImg = srcImg.clone();
    cv::Mat result = srcImg.clone();
    int ret = CCLDetect(testImg);
    if (ret > 0)
    {
        mvResult* pRes = (mvResult*)&pAlg->result;
        //cclTtemsDraw(pRes->cc_items, result);
        DrawResult(pAlg, result);
    }
    namedWindow("result", WINDOW_FULLSCREEN);
    cv::imshow("result", result);

    key = cv::waitKey(0);
    while (key != 'q' && key != 'Q')
    {
        cv::waitKey(0);
    }
    return 0;
}
