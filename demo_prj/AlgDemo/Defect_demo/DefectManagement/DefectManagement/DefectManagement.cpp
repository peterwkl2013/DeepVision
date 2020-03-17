#include "DefectManagement.h"
#include <QImage>


DefectManagement::DefectManagement(QWidget *parent)
    : QWidget(parent), model(nullptr)
{
    ui = new Ui::DefectManagementClass;
    ui->setupUi(this);
    pAlg = NULL;
    qRegisterMetaType<QList<QPersistentModelIndex>>("QList<QPersistentModelIndex>");
    qRegisterMetaType<QVector<int>>("QVector<int>");
    qRegisterMetaType<QAbstractItemModel::LayoutChangeHint>("QAbstractItemModel::LayoutChangeHint");
    InitUI();
}

DefectManagement::~DefectManagement()
{
    delete ui;
}

void DefectManagement::InitUI()
{
    connect(ui->btnClose, &QPushButton::clicked, this, &DefectManagement::close);
    connect(ui->btnDefect, &QPushButton::clicked, this, &DefectManagement::ExecuteDefectDetect);
    std::string tmpConfPath = "F:/AlgDemo/3rdparty/imvcfg/";
    this->maxWidth = 1024;
    this->maxHeight = 750;
    this->scale = 1;
    SetConfigPath(tmpConfPath);
    InitAlgParam();

    model = new QStandardItemModel(ui->treeView);
    connect(ui->treeView, &QTreeView::clicked, this, &DefectManagement::OnTreeViewItemClick);
    connect(this, &DefectManagement::SigReDrawResult, this, &DefectManagement::OnReDrawDefectResult);
    model->setHorizontalHeaderLabels(QStringList() << QStringLiteral("缺陷列表") << QStringLiteral("信息"));
    ui->treeView->setModel(model);
    ui->treeView->setColumnWidth(0, 140);
    checkedIndex.clear();
    pAlg = (algDllHandle*)mvInstanceAlloc(this->maxWidth * this->scale, this->maxHeight * this->scale, MV_ALG_DEFECTSDET, &param);
    if (!pAlg)
    {
        qDebug() << "err";
    }
    pAlg->alg_params.disp_level |= (0x01 << 21);
    //pAlg->algParam.disLevel |= (0x01 << 22);
    pAlg->alg_params.disp_level &= ~(0x01 << 22);
}

void DefectManagement::SetConfigPath(const std::string & path)
{
    this->configPath = path;
}

void DefectManagement::InitAlgParam()
{
    strcpy_s(this->param.cfg_path, this->configPath.c_str());
    this->param.online_temp = 0;    //not used
    this->param.tmp_img.pframe = NULL;
    this->param.tmp_img.width = 0;        //temp->width;
    this->param.tmp_img.height = 0;       //temp->height;
    this->param.tmp_img.channels = 0;     //temp->nChannels;
    this->param.tmp_img.depth = 0;        //temp->depth;
    this->param.tmp_img.wstep = 0;    //temp->widthStep;
    this->param.tmp_img.type = MV_CV_IPL;        //MV_CV_IPL;
}

bool DefectManagement::DefectDetectAndRotate(const Mat & tmpl, const Mat & testPic, algDllHandle * pAlg, const int & width, const int & height) const
{
    if (!pAlg)
    {
        return false;
    }
    mvResult* pRes = (mvResult*)&pAlg->result;
    /* 算法结果输出 */
    //eval = (defectsUserResult*)pRes->usrDat;
    mvInputImage orgImage;
    /////////////////////////////////////template image////////////////
    Mat tmpTmpl;
    cv::resize(tmpl, tmpTmpl, Size(width, height));
    cvtColor(tmpTmpl, tmpTmpl, cv::COLOR_GRAY2BGR, 0);
    Mat frame = tmpTmpl.clone();
    orgImage.index = 0;           /* index = 0,  template */
    orgImage.pframe = (void*)frame.data;        /* 图像数据地址 */
    orgImage.width = frame.cols;                /* 图像宽度 */
    orgImage.height = frame.rows;               /* 图像高度 */
    orgImage.channels = frame.channels();       /* 图像通道*/
    orgImage.wstep = frame.cols * frame.channels();  /* 图像 widthStep = 宽度* 通道数 */
    orgImage.depth = frame.depth();         /* 图像深度 */
    orgImage.type = MV_BGR24;   /*帧的格式*/
    int ret = mvAlgProcess(pAlg, (mvInputImage*)&orgImage);
    /////////////////////////////////////test image//////////////////////////////////////
    if (ret > 0)
    {
        Mat tmpTestPic;
        cv::resize(testPic, tmpTestPic, Size(width, height));
        cvtColor(tmpTestPic, tmpTestPic, cv::COLOR_GRAY2BGR, 0);
        Mat testPicCopy = tmpTestPic.clone();
        mvInputImage testImage;
        testImage.index = 1;           /* index = 0,  template, */
        testImage.pframe = (void*)testPicCopy.data;        /* 图像数据地址 */
        testImage.width = testPicCopy.cols;                /* 图像宽度 */
        testImage.height = testPicCopy.rows;               /* 图像高度 */
        testImage.channels = testPicCopy.channels();       /* 图像通道*/
        testImage.wstep = testPicCopy.cols * testPicCopy.channels();  /* 图像 widthStep = 宽度* 通道数 */
        testImage.depth = testPicCopy.depth();         /* 图像深度 */
        testImage.type = MV_BGR24;   /*帧的格式*/
        ret = mvAlgProcess(pAlg, (mvInputImage*)&testImage);
        if (ret > 0)
        {
            return true;
        }
        return false;
    }
    return false;
}

void DefectManagement::ShowImage(Mat& srcImg)
{
    QImage orgImg = QImage(srcImg.data, srcImg.cols, srcImg.rows, QImage::Format_RGB888);
    QImage scaledImg = orgImg.scaled((int)(((srcImg.cols / scale) / 8) * 8),
        (int)(((srcImg.rows / scale) / 8) * 8), Qt::KeepAspectRatio);
    QPixmap pixmapImg = QPixmap::fromImage(scaledImg);
    ui->ImageLabel->setPixmap(pixmapImg);
    ui->ImageLabel->setAlignment(Qt::AlignCenter);
}

void DefectManagement::Defect()
{
    mutex.lock();
    Mat templateImg = imread("E:/AlgDemo/3rdparty/pic/bd1.bmp", IMREAD_GRAYSCALE);
    Mat testOrg = imread("E:/AlgDemo/3rdparty/pic/bd2.bmp", IMREAD_GRAYSCALE);
    this->outDefectPic = testOrg.clone();
    if (DefectDetectAndRotate(templateImg, testOrg, pAlg, this->maxWidth * this->scale, this->maxHeight * this->scale))
    {
        cv::resize(this->outDefectPic, this->outDefectPic, cv::Size(this->maxWidth * this->scale, this->maxHeight * this->scale));
        pRes = (mvResult*)&pAlg->result;
        DrawDefectResult(pRes, outDefectPic);
        UpdateTreeView();
    }
    mutex.unlock();

}

void DefectManagement::ExecuteDefectDetect()
{
    std::thread t1(&DefectManagement::Defect, this);
    t1.detach();
}

void DefectManagement::DrawDefectResult(mvResult* pRes, Mat& img)
{
    if (!pRes)
    {
        return;
    }
    cv::cvtColor(img, img, COLOR_GRAY2RGB);
    vecMatchObj.clear();
    defectsUserResult* eval = (defectsUserResult*)pRes->user_dat;

    qDebug() << eval->angle;
    int lineLenght = 10;
    if (pRes->mat_objs.num_valid > 0)
    {
        for (auto i = 0; i < pRes->mat_objs.num_obj; i++)
        {
            matchObj *moj = &pRes->mat_objs.mat_obj[i];
            vecMatchObj.push_back(moj);
            vector<cv::Point> pnts;

            for (int i = 0; i < 4; i++)
            {
                pnts.push_back(cv::Point(moj->rot_box.pnts[i].x, moj->rot_box.pnts[i].y));
            }
            cv::polylines(img, pnts, true, Scalar(255, 0, 0), 2);
            cv::Point centerPoint = cv::Point(moj->center.x, moj->center.y);
            cv::line(img, centerPoint, cv::Point(centerPoint.x, centerPoint.y - lineLenght), Scalar(0, 255, 0), 2);
            cv::line(img, centerPoint, cv::Point(centerPoint.x + lineLenght, centerPoint.y), Scalar(0, 255, 0), 2);
            cv::line(img, centerPoint, cv::Point(centerPoint.x, centerPoint.y + lineLenght), Scalar(0, 255, 0), 2);
            cv::line(img, centerPoint, cv::Point(centerPoint.x - lineLenght, centerPoint.y), Scalar(0, 255, 0), 2);
            cv::line(img, centerPoint, centerPoint, Scalar(255, 0, 0), 3);
        }
    }
    //cv::imwrite("D:/img/resize.bmp", img);
    mvTransform2DImage((unsigned char*)img.data, img.cols, img.rows, img.channels(), (double *)eval->hmat);
    ShowImage(img);
}

void DefectManagement::DrawDefectResult(mvResult* pRes, double scale, const Mat& orgImg, Mat& outImg)
{
    if (!pRes)
    {
        return;
    }
    cv::Mat img = orgImg.clone();
    cv::cvtColor(img, img, cv::COLOR_GRAY2RGB);
    vecMatchObj.clear();
    defectsUserResult* eval = (defectsUserResult*)pRes->user_dat;
    //mvTransform2DImage((unsigned char*)img.data, img.cols, img.rows, img.channels(), eval->hmat);
    int lineLenght = 10;
    if (pRes->mat_objs.num_valid > 0)
    {
        for (auto i = 0; i < pRes->mat_objs.num_obj; i++)
        {
            matchObj *moj = &pRes->mat_objs.mat_obj[i];
            vecMatchObj.push_back(moj);
            vector<cv::Point> pnts;
            for (int i = 0; i < 4; i++)
            {
                pnts.push_back(cv::Point(moj->rot_box.pnts[i].x, moj->rot_box.pnts[i].y));
            }
            cv::polylines(img, pnts, true, Scalar(0, 0, 255), 2);
            cv::Point centerPoint = cv::Point(moj->center.x, moj->center.y);
            cv::line(img, centerPoint, cv::Point(centerPoint.x, centerPoint.y - lineLenght), Scalar(0, 255, 0), 2);
            cv::line(img, centerPoint, cv::Point(centerPoint.x + lineLenght, centerPoint.y), Scalar(0, 255, 0), 2);
            cv::line(img, centerPoint, cv::Point(centerPoint.x, centerPoint.y + lineLenght), Scalar(0, 255, 0), 2);
            cv::line(img, centerPoint, cv::Point(centerPoint.x - lineLenght, centerPoint.y), Scalar(0, 255, 0), 2);
            cv::line(img, centerPoint, centerPoint, Scalar(255, 0, 0), 3);
        }
    }
    mvTransform2DImage((unsigned char*)img.data, img.cols, img.rows, img.channels(), (double *)eval->hmat);
    outImg = img;
    //imwrite("D:/img/xxx.bmp", img);
    ShowImage(img);
}

void DefectManagement::ReDrawDefectResult(mvResult* pRes, Mat& img)
{
    if (!pRes)
    {
        return;
    }
    /*cv::cvtColor(img, img, CV_GRAY2RGB);
    vecMatchObj.clear();
    defectsUserResult* eval = (defectsUserResult*)pRes->usrDat;
    mvTransorm2DImage((unsigned char*)img.data, img.cols, img.rows, img.channels(), eval->hmat);*/
    int lineLenght = 10;
    if (pRes->mat_objs.num_valid > 0)
    {
        for (auto i = 0; i < pRes->mat_objs.num_obj; i++)
        {
            matchObj *moj = &pRes->mat_objs.mat_obj[i];
            vecMatchObj.push_back(moj);
            vector<cv::Point> pnts;
            for (int i = 0; i < 4; i++)
            {
                pnts.push_back(cv::Point(moj->rot_box.pnts[i].x, moj->rot_box.pnts[i].y));
            }
            cv::polylines(img, pnts, true, Scalar(0, 0, 255), 2);
            cv::Point centerPoint = cv::Point(moj->center.x, moj->center.y);
            cv::Scalar scalar;
            auto it = std::find(checkedIndex.begin(), checkedIndex.end(), i);
            if (it != checkedIndex.end())
            {
                scalar = cv::Scalar(0, 255, 0);
            }
            else
            {
                scalar = cv::Scalar(0, 100, 100);
            }
            cv::line(img, centerPoint, cv::Point(centerPoint.x, centerPoint.y - lineLenght), scalar, 2);
            cv::line(img, centerPoint, cv::Point(centerPoint.x + lineLenght, centerPoint.y), scalar, 2);
            cv::line(img, centerPoint, cv::Point(centerPoint.x, centerPoint.y + lineLenght), scalar, 2);
            cv::line(img, centerPoint, cv::Point(centerPoint.x - lineLenght, centerPoint.y), scalar, 2);
            cv::line(img, centerPoint, centerPoint, cv::Scalar(255, 0, 0), 3);
        }
    }
    ShowImage(img);
}

void DefectManagement::ReDrawDefectResult(Mat& img)
{
    int lineLenght = 10;
    for (auto i = 0; i < vecMatchObj.size(); i++)
    {
        matchObj *moj = vecMatchObj[i];
        auto it = std::find(checkedIndex.begin(), checkedIndex.end(), i);
        vector<cv::Point> pnts;
        for (int i = 0; i < 4; i++)
        {
            pnts.push_back(cv::Point(moj->rot_box.pnts[i].x, moj->rot_box.pnts[i].y));
        }
        cv::polylines(img, pnts, true, cv::Scalar(255, 0, 0), 2);
        cv::Point centerPoint = cv::Point(moj->center.x, moj->center.y);
        cv::Scalar scalar;
        if (it != checkedIndex.end())
        {
            scalar = cv::Scalar(0, 255, 0);
        }
        else
        {
            scalar = cv::Scalar(0, 100, 100);
        }
        cv::line(img, centerPoint, cv::Point(centerPoint.x, centerPoint.y - lineLenght), scalar, 2);
        cv::line(img, centerPoint, cv::Point(centerPoint.x + lineLenght, centerPoint.y), scalar, 2);
        cv::line(img, centerPoint, cv::Point(centerPoint.x, centerPoint.y + lineLenght), scalar, 2);
        cv::line(img, centerPoint, cv::Point(centerPoint.x - lineLenght, centerPoint.y), scalar, 2);
        cv::line(img, centerPoint, centerPoint, cv::Scalar(255, 0, 0), 3);
    }
    ShowImage(img);
}

void DefectManagement::UpdateTreeView()
{
    if (vecMatchObj.size() == 0)
    {
        return;
    }
    for (size_t i = 0; i < vecMatchObj.size(); i++)
    {
        QString itemProjectStr = QStringLiteral("缺陷目标%1").arg(i + 1);
        QStandardItem* itemProject = new QStandardItem(itemProjectStr);
        itemProject->setCheckable(true);
        model->appendRow(itemProject);
        model->setItem(model->indexFromItem(itemProject).row(), 1, new QStandardItem(QStringLiteral("缺陷信息说明")));
        QStandardItem* itemChild1 = new QStandardItem(QStringLiteral("坐标位置"));
        itemProject->appendRow(itemChild1);
        QString asix = QString("(%1,%2)").arg(vecMatchObj[i]->center.x).arg(vecMatchObj[i]->center.y);
        itemProject->setChild(itemChild1->index().row(), 1, new QStandardItem(asix));
        QStandardItem* itemChild2 = new QStandardItem(QStringLiteral("缺陷面积"));
        itemProject->appendRow(itemChild2);
        itemProject->setChild(itemChild2->index().row(), 1, new QStandardItem(QStringLiteral("%1").arg(vecMatchObj[i]->rot_box.area, 15, 'f', 2)));
        QStandardItem* itemChild3 = new QStandardItem(QStringLiteral("旋转角度"));
        itemProject->appendRow(itemChild3);
        itemProject->setChild(itemChild3->index().row(), 1, new QStandardItem(QStringLiteral("%1").arg(vecMatchObj[i]->rot_box.angle, 8, 'f', 4)));
        QStandardItem* itemChild4 = new QStandardItem(QStringLiteral("缺陷点YY"));
        itemProject->appendRow(itemChild4);
        itemProject->setChild(itemChild4->index().row(), 1, new QStandardItem(QStringLiteral("信息说明2")));
    }
}


void DefectManagement::OnTreeViewItemClick(const QModelIndex &index)
{
    QStandardItem* currentItem = model->itemFromIndex(index);
    qDebug() << index.data().toString();
    if (currentItem->isCheckable())
    {
        Qt::CheckState status = currentItem->checkState();
        if (status == Qt::Checked)
        {
            checkedIndex.push_back(currentItem->index().row());
        }
        else
        {
            if (checkedIndex.empty())
            {
                return;
            }
            auto it = std::find(checkedIndex.begin(), checkedIndex.end(), currentItem->index().row());
            if (it != checkedIndex.end())
            {
                checkedIndex.remove(currentItem->index().row());
            }
        }
        /*for (auto it = checkedIndex.begin(); it != checkedIndex.end(); it++)
        {
            qDebug() << *it;
        }*/
        emit SigReDrawResult();
    }
}

void DefectManagement::OnReDrawDefectResult()
{
    ReDrawDefectResult(pRes, outDefectPic);
}
