#include "extern_header.h"

const std::string path1 = R"(e:\AlgDemo\3rdparty\pic\aoi\bd2_750x1024.bmp)";
const std::string path2 = R"(e:\AlgDemo\3rdparty\pic\aoi\bd2_750x1024.bmp)";

//const std::string path1 = R"(E:\pic\6.bmp)";
//const std::string path2 = R"(E:\pic\7.bmp)";

cv::Mat tmpl_image;
cv::Mat test_image;
cv::Mat tmpl_roi_img;
cv::Point tmpl_offset(341, 259);
//cv::Point tmpl_offset(1330, 900);
std::string config_path = "e:/AlgDemo/3rdparty/imvcfg/";
std::vector<cv::Mat> vec_tmpl_images;
std::vector<cv::Mat> vec_test_images;
std::vector<cv::Point2f> vec_offset_pnt;
std::vector<std::string> vec_class_id;
mv2DTemplateMatch m_2d_tmpl_match;

algDllHandle *pAlg = nullptr;
initParam param;

double matrix3x3[9] = { 0 };

void InitAlgConfig()
{
    strcpy_s(param.cfg_path, config_path.c_str());
    //param.tmp_offset        = offset;
    //param.tmp_roi           = loc;
    param.online_temp = 0;    //not used
    param.tmp_img.pframe = NULL;
    param.tmp_img.width = 0;        //temp->width;
    param.tmp_img.height = 0;       //temp->height;
    param.tmp_img.channels = 0;     //temp->nChannels;
    param.tmp_img.depth = 0;        //temp->depth;
    param.tmp_img.wstep = 0;    //temp->wstep;
    param.tmp_img.type = MV_BGR24;        //MV_CV_IPL;
}

bool AllocateAlg()
{
    pAlg = (algDllHandle*)mvInstanceAlloc(tmpl_image.cols, tmpl_image.rows, MV_ALG_DEFECTSDET2, &param);
    if (pAlg == nullptr)
    {
        return false;
    }
    pAlg->alg_params.disp_level |= (0x01 << 21);
    pAlg->alg_params.disp_level |= (0x01 << 22);
    //pAlg->alg_params.disp_level &= ~(0x01 << 22);
    return true;
}

bool ExecuteAlg()
{
    cv::Mat frame = tmpl_image.clone();
    cv::Mat testCopy = test_image.clone();

    mvTempImg temp;
    mvInputImage orgImage;
    orgImage.index = 0;           /* index = 0,  template */
    orgImage.pframe = (void*)frame.data;        /* 图像数据地址 */
    orgImage.width = frame.cols;                /* 图像宽度 */
    orgImage.height = frame.rows;               /* 图像高度 */
    orgImage.channels = frame.channels();       /* 图像通道*/
    orgImage.wstep = frame.cols * frame.channels();  /* 图像 wstep = 宽度* 通道数 */
    orgImage.depth = frame.depth();         /* 图像深度 */
    orgImage.type = MV_BGR24;   /*帧的格式*/
    int64 startTime = cv::getTickCount();
    int ret = mvAlgProcess(pAlg, (mvInputImage*)&orgImage);
    double endTime = (cv::getTickCount() - startTime) * 1000. / cv::getTickFrequency();
    cout << "template image det time = " << endTime << "ms" << endl;
    if (ret > 0)
    {
        mvInputImage testImage;
        testImage.index = 1;       /* index = 0,  template, */
        testImage.pframe = (void*)testCopy.data;        /* 图像数据地址 */
        testImage.width = testCopy.cols;                /* 图像宽度 */
        testImage.height = testCopy.rows;               /* 图像高度 */
        testImage.channels = testCopy.channels();       /* 图像通道*/
        testImage.wstep = testCopy.cols * testCopy.channels();  /* 图像 wstep = 宽度* 通道数 */
        testImage.depth = testCopy.depth();         /* 图像深度 */
        testImage.type = MV_BGR24;   /*帧的格式*/

                                     /* 把转换矩阵数据传入*/
        temp.temp.index = 0;
        temp.temp.pframe = (void*)frame.data;
        temp.temp.width = frame.cols;
        temp.temp.height = frame.rows;
        temp.temp.channels = frame.channels();
        testImage.nsize = frame.cols * testCopy.rows * testCopy.channels();
        temp.temp.wstep = frame.cols * testCopy.channels();
        temp.temp.depth = frame.depth();
        temp.temp.type = MV_BGR24;
        temp.type = 0;//0-2*3 1-3*3 2-4*4

                      /* 传递变换matrix数据*/
        for (int i = 0; i < 9; i++)
        {
            temp.tran_matrix[i] = matrix3x3[i];
        }
        testImage.user_param = (void*)&temp;

        startTime = cv::getTickCount();
        ret = mvAlgProcess(pAlg, (mvInputImage*)&testImage);
        endTime = (cv::getTickCount() - startTime) * 1000. / cv::getTickFrequency();
        cout << "test image det time = " << endTime << " ms" << endl;
        if (ret > 0)
        {
            return true;
            mvResult* pRes = (mvResult*)&pAlg->result;
            mvMatchObjsDrawAndDisplay(pAlg, pRes);
            imshow("det-result", testCopy);
        }
    }
    return false;
}

void DrawCross(cv::Mat & img, vector<cv::Point>& pnts, cv::Point centerPoint, const cv::Scalar & lineColor1, const cv::Scalar & lineColor2, const int & lineLength)
{
    cv::polylines(img, pnts, true, lineColor1, 2);
    cv::line(img, centerPoint, cv::Point(centerPoint.x, centerPoint.y - lineLength), lineColor2, 2);
    cv::line(img, centerPoint, cv::Point(centerPoint.x + lineLength, centerPoint.y), lineColor2, 2);
    cv::line(img, centerPoint, cv::Point(centerPoint.x, centerPoint.y + lineLength), lineColor2, 2);
    cv::line(img, centerPoint, cv::Point(centerPoint.x - lineLength, centerPoint.y), lineColor2, 2);
    cv::line(img, centerPoint, centerPoint, cv::Scalar(0, 0, 255), 3);
}

void DrawResult(cv::Mat & img)
{
    if (!pAlg || !img.data)
    {
        return;
    }
    mvResult* pRes = (mvResult*)&pAlg->result;
    defectsUserResult* eval = (defectsUserResult*)pRes->user_dat;
    int lineLenght = 10;
    for (auto i = 0; i < pRes->mat_objs.num_obj; i++)
    {
        matchObj moj = pRes->mat_objs.mat_obj[i];
        //defectsFeature fea = eval->matObjFeature[i];
        vector<cv::Point> pnts;
        mvFPoint inPnt, outPnt;
        for (int i = 0; i < 4; i++)
        {
            inPnt.x = moj.rot_box.pnts[i].x;
            inPnt.y = moj.rot_box.pnts[i].y;
            pnts.emplace_back(cv::Point(inPnt.x, inPnt.y));
        }
        inPnt.x = moj.center.x;
        inPnt.y = moj.center.y;
        cv::Point centerPoint = cv::Point(inPnt.x, inPnt.y);
        DrawCross(img, pnts, centerPoint, cv::Scalar(0, 0, 255), cv::Scalar(0, 255, 0), lineLenght);
    }
}


void InitConfig()
{
    tmpl_image = cv::imread(path1, IMREAD_ANYCOLOR | IMREAD_ANYDEPTH);
    test_image = cv::imread(path2, IMREAD_ANYCOLOR | IMREAD_ANYDEPTH);
    if (!tmpl_image.data || !test_image.data)
    {
        cout << "Load image fail" << endl;
        return;
    }
    tmpl_roi_img = tmpl_image(Rect(tmpl_offset.x, tmpl_offset.y, 160, 200));
    //tmpl_roi_img = tmpl_image(Rect(tmpl_offset.x, tmpl_offset.y, 1000, 1210));
    vec_tmpl_images.emplace_back(tmpl_image);
    vec_test_images.emplace_back(test_image);
    vec_offset_pnt.emplace_back(tmpl_offset);
    vec_class_id.emplace_back(std::string("0"));
    m_2d_tmpl_match.config_path = config_path;
    m_2d_tmpl_match.num_detect = 1;
    m_2d_tmpl_match.filter_size = 5;
    m_2d_tmpl_match.num_features = 64;
}

bool TrainTemplate()
{
    int ret = m_2d_tmpl_match.trainTemplate(vec_tmpl_images, vec_class_id, vec_offset_pnt);
    if (ret <= 0)
    {
        return false;
    }
    return true;
}

bool MatchTemplate()
{
    vector<matchResult> mres;
    m_2d_tmpl_match.matchTemplate(vec_test_images[0].clone(), vec_class_id, mres, 90);
    if (mres.size() == 0)
    {
        return false;
    }
    matchResult obj = mres[0];
    m_2d_tmpl_match.getAFTransform(obj, matrix3x3);
    for (int i = 0; i < 9; i++)
    {
        cout << "matrix3x3[" << i << "]=" << matrix3x3[i] << endl;
    }
    return true;
}

bool PrecisionPosition(const cv::Mat & det_img, std::vector<cv::Point2f>& mask_points_tmpl, std::vector<cv::Point2f>& mask_points_det)
{
    if (det_img.empty() || mask_points_tmpl.size() == 0 || mask_points_det.size() == 0)
        return false;



}

int main()
{
    InitConfig();
    if (!TrainTemplate())
    {
        cout << "train template fail";
        return -1;
    }
    double matrix3x3[9]{ 0 };
    if (!MatchTemplate())
    {
        cout << "match template fail";
        return -1;
    }
    InitAlgConfig();
    if (!AllocateAlg())
    {
        cout << "allocate algrithm fail" << endl;
        return -1;
    }
    if (!ExecuteAlg())
    {
        cout << "exec alg fail";
        return -1;
    }
    cv::Mat out_img = test_image.clone();
    DrawResult(out_img);
    cv::imshow("defect result", out_img);
    cv::imwrite("D:/img/result.bmp", out_img);
    cv::waitKey();
    return 0;
}
