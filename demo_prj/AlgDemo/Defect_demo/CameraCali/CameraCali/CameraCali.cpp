#include "CameraCali.h"
#include <QPushButton>

#include <QDebug>

int CameraCali::counter = 0;


CameraCali::CameraCali(QWidget *parent)
    : QWidget(parent), pAlg(nullptr), flag(false), pRes(nullptr)
{
    ui.setupUi(this);

    ui.label_img->resize(640, 480);
    connect(ui.btn_start, &QPushButton::clicked, this, &CameraCali::OnStartBtnClicked);
    connect(ui.btn_stop, &QPushButton::clicked, this, &CameraCali::OnStopBtnClicked);
    connect(this, &CameraCali::SigShowImage, this, &CameraCali::OnShowImage);
    //init param;
    Init();
}

void CameraCali::Init()
{
    if (!InitParam())
        return;
    if (!AllocPalg())
        return;
    if (!UserParamSetting())
        return;
}

void CameraCali::OnStartBtnClicked()
{
    flag = true;
    if (cap.isOpened())
    {
        return;
    }
    cap.open(0);
    if (!cap.isOpened())
    {
        cap.open(1);
    }
    int num = 10;
    while (flag)
    {
        cap >> srcImage;
        /*char name[20] = { 0 };
        sprintf_s(name, "D:/img/img_%d.bmp", num);
        imwrite(name, srcImage);
        imshow("src", srcImage);
        */
        int ret = ExecAlg();
        qDebug("execute alg result=%d", ret);

        if (ret)
        {
            pRes = (mvResult*)&pAlg->result;
            mvCaliParamsEX *user_dat = (mvCaliParamsEX*)pRes->user_dat;
            /* 如果已经相机标定 */
            if (user_dat->calibrated)
            {
                mvFPoint pp1, pp2;
                double d;
                vector<double>dd;
                vector <mvFPoint> pnts1, pnts2;

                ex_param = *user_dat;
                pnts1.clear();
                pnts2.clear();
                pp1.x = 100, pp1.y = 100;
                pp2.x = 200, pp2.y = 200;
                /* 测量两点之间物理距离 */
                mvCali3DDistance(&ex_param, pp1, pp2, d);
                qDebug("distance = %f mm", d);
                pnts1.push_back(pp1);
                pnts2.push_back(pp2);
                pp1.x = 20;  pp1.y = 20;
                pp2.x = 100; pp2.y = 100;
                pnts1.push_back(pp1);
                pnts2.push_back(pp2);

                /* 测量点对之间物理距离 */
                mvCali3DDistance(&ex_param, pnts1, pnts2, dd);
                qDebug() << "distance:";
                for (int k = 0; k < dd.size(); k++)
                {
                    double t = dd[k];
                    qDebug("%f mm", t);
                }
                cout << endl;
            }
        }
        emit SigShowImage(srcImage);
        cv::waitKey(10);
    }
}

void CameraCali::OnStopBtnClicked()
{
    flag = false;
    if (cap.isOpened())
    {
        cap.release();
    }
}

bool CameraCali::InitParam()
{
    strcpy_s(param.cfg_path, MV_CFG_PATH);
    return true;
}

bool CameraCali::AllocPalg()
{
    pAlg = (algDllHandle*)mvInstanceAlloc(IMG_WIDTH, IMG_HEIGHT, MV_ALG_MONO_CAM_CALI, &param);
    if (!pAlg)
        return false;
    return true;
}

bool CameraCali::UserParamSetting()
{
    if (!pAlg)
        return false;
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
    return true;
}

int CameraCali::ExecAlg()
{
    if (!pAlg || !srcImage.data)
    {
        return 0;
    }
    lroi.up_left.x = 0;  lroi.dw_right.x = srcImage.cols;
    lroi.up_left.y = 0;  lroi.dw_right.y = srcImage.rows;
    detImage.index = counter++;
    detImage.pframe = (void*)srcImage.data;
    detImage.width = srcImage.cols;
    detImage.wstep = srcImage.step;
    detImage.height = srcImage.rows;
    detImage.channels = srcImage.channels();
    detImage.depth = srcImage.depth();
    detImage.type = MV_BGR24;

    int ret = mvAlgProcess(pAlg, (mvInputImage*)&detImage);

    return ret;
}

void CameraCali::OnShowImage(const Mat & org)
{
    if (!org.data)
    {
        return;
    }
    Mat img = org.clone();
    cvtColor(img, img, cv::COLOR_BGR2RGB);
    QImage orgImg = QImage(img.data, img.cols, img.rows, QImage::Format_RGB888);
    QImage scaledImg = orgImg.scaled((int)((img.cols / 8) * 8),
        (int)((img.rows / 8) * 8), Qt::KeepAspectRatio);
    QPixmap pixmapImg = QPixmap::fromImage(scaledImg);
    ui.label_img->setPixmap(pixmapImg);
    ui.label_img->setAlignment(Qt::AlignCenter);
}
