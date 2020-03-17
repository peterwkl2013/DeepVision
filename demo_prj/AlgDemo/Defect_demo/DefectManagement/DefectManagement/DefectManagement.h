#pragma once

#include <iostream>
#include <QtWidgets/QWidget>
#include "ui_DefectManagement.h"
#include <QPushButton>
#include <opencv2\opencv.hpp>
#include "../../../3rdparty/DllMvInterface.h"
#include <Qdebug>
#include <vector>
#include <QStandardItemModel>
#include <QStandardItem>
#include <QModelIndex>
#include <QDebug>
#include <list>
#include <QTreeView>
#include <thread>
#include <mutex>
#include <exception>
#include <QMetaType>


using namespace cv;
using namespace std;


typedef struct
{
    mvRGBImage tmpImg;       //template img
    mvRGBImage curImg;       //current process img
    float hmat[9];           //3x3 transorm mat
    float angle;             //rotate angle
    int reserved1;
    float reservd2;
}defectsUserResult;


class DefectManagement : public QWidget
{
    Q_OBJECT

public:
    DefectManagement(QWidget *parent = Q_NULLPTR);
    ~DefectManagement();
    void InitUI();
    void ExecuteDefectDetect();
    void DrawDefectResult(mvResult* pRes, Mat& img);
    void DrawDefectResult(mvResult* pRes, double scale, const Mat& orgImg, Mat& outImg);
    void ReDrawDefectResult(mvResult * pRes, Mat & img);
    void ReDrawDefectResult(Mat & img);
    void UpdateTreeView();
    void SetConfigPath(const std::string& path);
    void InitAlgParam();
    bool DefectDetectAndRotate(const Mat & tmpl, const Mat & testPic, algDllHandle * pAlg, const int & width, const int & height) const;
    void ShowImage(Mat & srcImg);
    void Defect();
    void OnTreeViewItemClick(const QModelIndex &index);
    void OnReDrawDefectResult();
signals:
    void SigReDrawResult();

private:
    Ui::DefectManagementClass *ui;
    std::string configPath;
    initParam param;
    int maxWidth;
    int maxHeight;
    int scale;
    vector<matchObj *> vecMatchObj;
    QStandardItemModel *model;
    list<int> checkedIndex;
    Mat outDefectPic;
    mvResult* pRes;
    algDllHandle* pAlg;
    std::mutex mutex;
};
