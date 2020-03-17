/*****************************************************

object moving detection demo

Author:
wsn@20181025
******************************************************/

#include <cstdio>
#include <cstdlib>
#include <opencv2/opencv.hpp>
#include "../../3rdparty/DllMvInterface.h"
#include <opencv2/imgproc/imgproc_c.h>


using namespace cv;
using namespace std;

//
//#define MV_RELEASE2

#ifndef MV_RELEASE
#define MV_CFG_PATH "F:/AlgDemo/3rdparty/imvcfg/"
#else
#ifdef MV_RELEASE2
#define MV_CFG_PATH "./imvcfg/"
#else
#define MV_CFG_PATH "d:/imvcfg/"
#endif
#endif

#define MOUSE_TMP_WINDOW "testCamCalibration"

initParam param;

enum { MV_CHESSBOARD = 0, MV_CIRCLES_GRID = 1, MV_ASYMMETRIC_CIRCLES_GRID = 2 };
enum { MV_MONOCULAR_CAM = 0, MV_FISH_EYE = 1 };

typedef struct mvCaliSettings
{
    int   model_type;             /* intput, model_type monocular camera:0, fish-eye:1(not supported yet), default:0 */
    int   grid_type;              /* intput, chessbord:0, circles_grid:1, asysmmetric circles grid:2, default:0 */
    int   grid_width;             /* intput, grid width, default: 9 */
    int   grid_height;            /* intput, grid height, default: 6 */
    float grid_size;              /* intput, grid square size, default: 100 */
    int   num_frame;              /* intput, how many frames to calibration default: 25 */
    int   eps_count;              /* intput, terminate criteria count 80 */
    float eps;                    /* intput, terminate criteria eps, default: 0.05 */
    int   delay_time;             /* intput, internal delay time(ms) to process next frame, default: 100ms */
    int   flag;                   /* intput, flag for fish-eye model */
    int   aspect_ration;          /* intput, aspect ration flag, only for fish-eve model */
    int   zero_dist;              /* intput, zero tangent dist flag, only for fis-eye model */
    int   principal_point;        /* intput, fix principal point flag, only for fis-eye model */
    int   frame_cnt;              /* output, internal frame counter */
}mvCaliSettings;

typedef struct mvCaliParamsEX
{
    int     best_plane;           /* output, best plane with the min error */
    double  h2d3d[9];             /* output, homography matrix 2d-3d, distortion is taking account */
    double  h3d2d[9];             /* output, homography matrix 3d-2d, distortion is taking account */
    double  xyz_angle[3];         /* output, x, y, z rotaion angle */
    double  hcam[9];              /* output, camera's intrinsic matrix data */
    double  rot_mat[9];           /* output, rotation matrix */
    double  rvectors[9];          /* output, for rotation vectors */
    double  tvectors[9];          /* output, translation vectors */
    double  distkp[9];            /* output, k1,k2,p1,p2,k3 */
    double  min_err;              /* output, this plane error */
    int     num_point;            /* output, point num */
    double  total_err;            /* output, total pixels errors for total frame */
    int     calibrated;           /* output, is calibrated, true or false */
    mvCaliSettings cfg;           /* output, settings file config */
    char    output_path[512];     /* output, file path */
}mvCaliParamsEX;

extern int mvLoadCalibrationData(char *file_name, mvCaliParamsEX *param);
extern void mvCali3DDistance(mvCaliParamsEX *param, mvFPoint &pnt1, mvFPoint &pnt2, double &d);
extern void mvCali3DDistance(mvCaliParamsEX *param, vector<mvFPoint> &pnt1, vector<mvFPoint>&pnt2, vector<double> &d);
extern void mvCali2DDistance(mvCaliParamsEX *param, mvFPoint &pnt1, mvFPoint &pnt2, double &d);
extern void mvCali2DDistance(mvCaliParamsEX *param, vector<mvFPoint> &pnt1, vector<mvFPoint> &pnt2, vector<double> &d);
extern void mvImageUndistortion(mvCaliParamsEX *param, mvImage& img);

int main(int argc, char* argv[])
{
    int i, ret;
    char tmpStr[200];
    cv::Point pt;
    static int frameNum = 0;
    IplImage *pOutput;
    int index, flag = 0, initflag = 0;
    algDllHandle *pAlg = NULL;
    VideoCapture cap;
    mvPoint offset;
    mvRect loc;
    mvInputImage  detImage;
    Mat img;
    mvCaliSettings user_params;
    mvCaliParamsEX ex_param;

    printf("testMovingDetection@X-Vision SDK for Windows\n");
    printf("cmd usage: ./testX-Vision.exe n \nn: camera index, 0,1,2...\n");

    if (argc == 1)
        index = 1;
    else
        index = atoi(argv[1]);
    /* open camera */
    cap.open(index);
    if (!cap.isOpened())
        cap.open(0);
    if (!cap.isOpened())
    {
        cout << "unble to open the camera." << endl;
        return -1;
    }

    strcpy_s(param.cfg_path, MV_CFG_PATH);

    int delay = 10;
    mvRect lroi;
    static int counter = 0;
    mvFPoint pp1, pp2;
    double d;
    vector<double>dd;
    vector <mvFPoint> pnts1, pnts2;


    while (1)
    {
        cap >> img;
        if (frameNum++ < 5)
            continue;
        {
            if (flag == 0)
            {
                mvEngineCfg mvParam;

                flag = 1;

                pAlg = (algDllHandle*)mvInstanceAlloc(img.cols, img.rows, MV_ALG_MONO_CAM_CALI, &param);

                /* reset the parameters of alg input..., only setting the first time in the allocation step */
                /* if use the internal alg default param: pAlg->run_cfg.user_param is NULL */
                memset(&user_params, 0, sizeof(mvCaliSettings));
#if 1
                user_params.model_type = 0;
                user_params.grid_type = MV_CHESSBOARD;      //MV_ASYMMETRIC_CIRCLES_GRID
                user_params.grid_width = 9;
                user_params.grid_height = 6;
                user_params.grid_size = 100;    // 10 * 10 square meters per grid
#else
                user_params.grid_type = MV_ASYMMETRIC_CIRCLES_GRID;      //
                user_params.grid_width = 4;
                user_params.grid_height = 11;
                user_params.grid_size = 100;    // 10 * 10 square meters per grid
#endif
                user_params.num_frame = 25;
                user_params.eps_count = 80;     //
                user_params.eps = 0.05;
                user_params.delay_time = 100;    //100 ms delay to process next frame
                pAlg->run_cfg.user_param = &user_params;
            }

            {
                mvResult *pRes;


                lroi.up_left.x = 0;  lroi.dw_right.x = img.cols;
                lroi.up_left.y = 0;  lroi.dw_right.y = img.rows;

                detImage.index = counter++;
                detImage.pframe = (void*)img.data;
                detImage.width = img.cols;
                detImage.wstep = img.step;
                detImage.height = img.rows;
                detImage.channels = img.channels();
                detImage.depth = img.depth();
                detImage.type = MV_BGR24;


                int64 nTick;
                double ptime;
                nTick = getTickCount();
                ret = mvAlgProcess(pAlg, (mvInputImage*)&detImage);
                cout << "exec result=" << ret << endl;
                ptime = ((double)getTickCount() - nTick)*1000. / getTickFrequency();

                pRes = (mvResult*)&pAlg->result;
                if (ret)
                {
                    mvCaliParamsEX *user_dat = (mvCaliParamsEX*)pRes->user_dat;

                    imshow("real", img);

                    /* 如果已经相机标定 */
                    if (user_dat->calibrated)
                    {
                        //从本地读取相机标定数据
                        //char tmp_str[255];
                        //strcpy(tmp_str, MV_CFG_PATH);
                        //strcat(tmp_str, "mvcalid.dat");
                        //mvLoadCalibrationData(tmp_str, &ex_param);

                        ex_param = *user_dat;
                        pnts1.clear();
                        pnts2.clear();
                        pp1.x = 100, pp1.y = 100;
                        pp2.x = 200, pp2.y = 200;
                        /* 测量两点之间物理距离 */
                        mvCali3DDistance(&ex_param, pp1, pp2, d);
                        cout << "distance = " << d << "mm" << endl;
                        pnts1.push_back(pp1);
                        pnts2.push_back(pp2);
                        pp1.x = 20;  pp1.y = 20;
                        pp2.x = 100; pp2.y = 100;
                        pnts1.push_back(pp1);
                        pnts2.push_back(pp2);

                        /* 测量点对之间物理距离 */
                        mvCali3DDistance(&ex_param, pnts1, pnts2, dd);
                        cout << "distance:" << endl;
                        for (int k = 0; k < dd.size(); k++)
                        {
                            double t = dd[k];
                            cout << "" << t << "mm" << ",";
                        }
                        cout << endl;
                    }
                }

                cv::waitKey(10);
            }

            if (flag == 0)
            {
                mvInstanceDelete(pAlg);
                counter = 0;
                printf("succussfully destroy alg handle!\n");
            }
        }

    }

    cout << "program end. \n press anykey to continue.\n" << endl;
    getchar();

    return 0;
}
