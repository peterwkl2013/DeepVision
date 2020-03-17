#pragma once

#include <QtWidgets/QWidget>
#include "ui_CameraCali.h"
#include <iostream>
#include <opencv2\opencv.hpp>
#include "../../../3rdparty/DllMvInterface.h"

using namespace std;
using namespace cv;


#define MV_CFG_PATH "F:/AlgDemo/3rdparty/imvcfg/"
#define IMG_WIDTH 640
#define IMG_HEIGHT 480

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


class CameraCali : public QWidget
{
    Q_OBJECT

public:
    CameraCali(QWidget *parent = Q_NULLPTR);
    void Init();

    void OnStartBtnClicked();
    void OnStopBtnClicked();

    bool InitParam();
    bool AllocPalg();
    bool UserParamSetting();
    int ExecAlg();

    void OnShowImage(const Mat& org);

signals:
    void SigShowImage(Mat img);

private:
    Ui::CameraCaliClass ui;
    initParam param;
    algDllHandle * pAlg;
    VideoCapture cap;
    Mat srcImage;
    bool flag;
    mvCaliSettings user_params;
    mvRect lroi;
    mvInputImage  detImage;
    mvCaliParamsEX ex_param;
    mvResult *pRes;
    static int counter;
};
