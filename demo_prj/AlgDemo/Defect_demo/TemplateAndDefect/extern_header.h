#define CV__ENABLE_C_API_CTORS // enable C API ctors (must be removed)


#include <opencv2/opencv.hpp>
#include <opencv2/core/types_c.h>
#include <opencv2/core/core_c.h>
#include "opencv2/highgui/highgui_c.h"
#include "opencv2/imgproc/imgproc_c.h"
#include "opencv2/video/video.hpp"
#include <vector>
#include <fstream>
#include "DllmvInterface.h"
#include "mvSDK_interface.h"

using namespace std;
using namespace cv;


struct mvTemplateFeature
{
    int x;         /* x offset */
    int y;         /* y offset */
    int label;     /* Quantization */
    mvTemplateFeature() : x(0), y(0), label(0) {}
    mvTemplateFeature(int _x, int _y, int _label) : x(_x), y(_y), label(_label) {};
};

typedef struct
{
    int width;
    int height;
    int x_offset;    /* feature offset x */
    int y_offset;    /* feature offset y */
    int pyramid_level;
    std::vector<mvTemplateFeature>features;
}mvTemplate;

typedef struct
{
    float scale;           /* scale factor */
    float angle_step;      /* angle step */
    int   padding;         /* padding size */
    Point2f tempInfo[4];   /* template location in tempalte image */
    Point2f objbox[4];     /* box of the template when angle = 0 */
    Point2f xy_offset;     /* template offset to the image */
    float width;           /* original width of template */
    float height;          /* original height of template */
    int num_features;      /* num features */
    int num_detected;      /* num have detect */
}TemplateExtraParam;

typedef std::map<String, TemplateExtraParam> TemplateExtraParamMap;

/* match result */

typedef struct
{
    Point2f pnts[4];         /* 4-point represent of this box */
    Point2f center;          /* center point of this box */
    float   angle;           /* angle of this box */
    float   width;           /* width */
    float   height;          /* height */
    float   area;            /* area of this box */
}mvFBox;

typedef struct
{
    String class_id;
    int index;       /* template index */
    float score;
    float angle;     /* angle of this matched object*/
    float scale;     /* scale factor */
    Point2f r_center;  /* rotation center */
    mvFBox bbox;
}matchResult;

class mv2DTemplateMatch
{
public:
    mv2DTemplateMatch()
    {
        scale_ranges.push_back(1.0);
        angle_ranges.push_back(0.0); angle_ranges.push_back(360);
        num_features = 128; min_thres = 50; max_thres = 90;
        angle_step = 1;  filter_size = 5; /* */
        num_detect = 1;
        config_path = "F:/AlgDemo/3rdparty/imvcfg";
        overlap_thres = 0.7;
    }
    mv2DTemplateMatch(vector<float>_scale,
        vector<float>_angle, int _num_features, int _min_thres, int _max_thres, float angle_step, int _filter_size,
        int _num_detect, char *_path) {
        scale_ranges = _scale;
        angle_ranges = _angle;
        num_features = _num_features; min_thres = _min_thres; max_thres = _max_thres;
        angle_step = angle_step;  filter_size = _filter_size;
        num_detect = _num_detect;
        config_path = _path;
        rotation_flag = 0;
    }

    ~mv2DTemplateMatch();
    /*
    @input, temps  ,color-image
    @input, class_ids class name
    @input, template offset int the template iamge
    */
    int trainTemplate(vector<Mat>& temps, vector<String>& class_ids, vector <Point2f> offset);
    void updateTemplates(void);
    /*
    @input, det_img  detection image,color-image
    @input, class_ids class name
    @input, conf  score confidence (0~100)
    @output, mres, match result
    @reutrn true or false
    */
    int matchTemplate(Mat& det_img, vector<cv::String>& class_ids, vector<matchResult>& mres, float conf = 85);
    /*
    @output, the hmat, get the template affine transform matrix.
    */
    void getAFTransform(matchResult& mres, double* hmat);
    /*
    @temp, input, template image,  gray image
    @dst,  input, detect image, gray image
    @offset, input,  offset, usually within the template location
    @angle_range, input, angle range(+/-), default: 5
    @angle_step,  input, angle step,  default: 0.2
    @xy_range,   input, windows(x, y) size,   default: 30
    @matrix3x3, ouput, the affine matrix2x3 in [hmat]
    */
    void reffinePosition(Mat& temp, Mat& dst, Point2f offset, int angle_range = 5, float angle_step = 0.2, int xy_range = 30);
    /*
    @tempp, input, template image points.
    @dstpp, input, detect image points, the size as the template.
    @type,  type = 0, the affine matrix: 2x3,  else the homnography matrix 3x3
    @matrix3x3, ouput, the affine matrix2x3 in [hmat]
    */
    void reffinePosition(vector<Point2f>&tempp, vector<Point2f>& dstpp, int type = 0);
    /*
    @tempp, input, template points in the template image.
    @dstpp, output, output points in the detection image.
    */
    void calcPointPosition(vector<Point2f>& tempp, vector<Point2f>& dstpp);


    //map(class_id , mvTemplate)
    std::map<String, std::vector<vector<mvTemplate>>> template_features;
    TemplateExtraParamMap class_param;

public:
    String config_path;              /* template feature to store or to read from */
    int num_detect;                  /* num object to detect in one image */
    vector<float> scale_ranges;      /* scale factor(0.8~1.5) for training,  now only support scale = 1.0 */
    vector<float> angle_ranges;      /* rotation angle (0~360) for training,  default(0,360) */
    int   num_features;              /* feature to extract in the template */
    float min_thres;                 /* to control the feature response */
    float max_thres;                 /* to control the feature response */
    float   angle_step;              /* angle step when train template */
                                     /* when the template size(width, height) is too large  , it is necessary
                                     to set the filter size: 3, 5, 7, 9,11, to filter the features, else set to 0 , when have enough features */
    int   filter_size;
    double hmat[16];                /* output, current transformation matrix for : 2x3, 3x3, 4x4 */
    float reffine_angle;            /* output, the reffined angle of current transform */
    int rotation_flag;              /* check template is rotation flag */
    Point2f rotation_center;        /* template rotaion center */
    float overlap_thres;            /* multi-object with the same class id remove threshold,  rect overap */
private:
    int test;
};


extern void mvExtractContour(Mat& src_img, vector<vector<mvPoint>> &ppt_det);
//extern void mvExtractContour(Mat& src_img, vector<vector<mvPoint>> &ppt_det, Mat& loc_mask, int edge_thres = 100);
extern void mvCalAffinePoints(vector<vector<mvPoint>>& temp, Point temp_center, Point dst_center, float angle, float scale, vector<vector<mvPoint>>& dst);

extern float mv2DPointMatch_ICP(vector<Point2f>& tmp, vector<Point2f>& dst, double* matri3x3, double& angle,
    Mat& image, int max_ilter = 100, float _error = 0.001);

/* fill polygon area
@input, pnts input points
@input, image size,
@input,output, mask_out if empty create 3 channel mask image, else the input mask
return;
*/

extern void mvFillMask(vector<Point>& pnts, cv::Size dsize, Mat& mask_out);



typedef struct defectsFeature
{/* match Objects */
    int   index;           /* object index */
    int   mat_index;       /* matched object index */
    int   valid;           /* valid flag */
    int   num_pixels;
    float mean_gray;
    float area;
    float wh_rate;
    float angle;
    float contrast;
    float homogeneity;
    float entropy;
    float energy;
    float correlation;
    int   reserved1;
    float reseved2;
}defectsFeature;

typedef struct
{
    mvRGBImage tmp_img;                         /* template imgage */
    mvRGBImage cur_img;                         /* current process imgage */
    double     hmat[9];                         /* 3x3 transform mat */
    float      angle;                           /* image rotation angle */
    std::vector<defectsFeature> matObjFeature;  /* defects feature */
    std::vector<cv::Point2f> temp_points;       /* template edge points */
}defectsUserResult;
