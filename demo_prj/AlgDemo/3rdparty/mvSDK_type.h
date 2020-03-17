/*************************************************************************
*
*                    X-Vision SDK type defines       
*
*
* File Discription: 
*       type defines
*
* Author: WSN@2018/10/17
*
**************************************************************************/

#ifndef  __MVSDK_TYPE_H__
#define  __MVSDK_TYPE_H__

#include <vector>
#include "mvSDK_type.h"


#define MV_ALG_SCL 1

using namespace std;

#ifdef __cplusplus
extern "C" {
#endif

#ifndef MV_FALSE
#define MV_FALSE 0
#endif
#ifndef MV_TRUE
#define MV_TRUE 1
#endif
	/* xVision SDK algorithm type */
	typedef enum
	{
		MV_ALG_SHAPE_MATCH = 0,                /* match process use shape match metho */
		MV_ALG_SHAPE_TMP,                      /* template process use shape match metho, for MV_ALG_SHAPE_MATCH */
		MV_ALG_FEATURE_LOC,                    /* location process use feature points metho */
		MV_ALG_LOC_MATCH,                      /* match process use feature points */
		MV_ALG_FEATURE_TMP = 4,                /* template process use feature points metho, for MV_ALG_LOC_MATCH */
		MV_ALG_QRDECODE_DET,                   /* decoe qr code,or ean code etc. */
		MV_ALG_ZJLOC_DET,                      /* zujian location */
		MV_ALG_REVERTING_IMAGE,                /* revertin image by template*/
		MV_ALG_SURFACE_DET,                    /* surface detection */
		MV_ALG_TANZUAN_DET,                    /* tanzuan detection */
		MV_ALG_DOCKRECOG,                      /* dock recognize */
		MV_ALG_TRACKING_TMP,                   /* template process use tracking metho */
		MV_ALG_OBJECT_TRACKING,                /* object tracking */
		MV_ALG_FACE_TRACKING,                  /* face tracking */
		MV_ALG_ZJSURFACE_DET,                  /* zujian surface detection */
		MV_ALG_CIR_DET = 15,                   /* circle detection */
		MV_ALG_LINE_DET,                       /* line detection */
		MV_ALG_FACE_DET,                       /* face detection && recognization */
		MV_ALG_OBJECT_MATCH,                   /* object match */
		MV_ALG_PHONE_RECYCLE,                  /* phone recycle system */
		MV_ALG_COMPONENT_DET1 = 20,            /* componet object detect */
		MV_ALG_COMPONENT_DET2,                 /* componet object detect */
		MV_ALG_PHONE_MEAUREMENT,               /* phone recycle system */
		MV_ALG_ZMCLOTH_DET1,                   /* zm cloth surface detect */
		MV_ALG_ZMCLOTH_DET2,                   /* zm cloth surface detect */
		MV_ALG_ZMCLOTH_DET_DL1,                /* deep-learning model 1 use ssd network */
		MV_ALG_ZMCLOTH_DET_DL2,                /* deep-learning model 2 use fcn network */
		MV_ALG_ZMCLOTH_DET_DL3,                /* deep-learning model 2 use fcn network */
		MV_ALG_ZMCLOTH_DET_CAFFE_SSD,          /* deep-leanning model use caffe-ssd network */
		MV_ALG_ZMCLOTH_DET_CAFFE_FCN,          /* deep-leanning model use caffe-fcn network */
		MV_ALG_SEGMENTATION_PSPNET,            /* deep-leanning model use caffe-pspnet network */
		MV_ALG_LZCOUNTER,                      /* object detect and counter */
		MV_ALG_IMAGE_QULITY_EVAL,              /* image quality evaluation  */
		MV_ALG_FOCUS_EVAL,                     /* image focus evaluation  */
		MV_ALG_DARKNET_PROCESS,                /* image quality evaluation  */
		MV_ALG_DEFECTSDET,                     /* image defects detection & revert image */
		MV_ALG_DEFECTSDET2,                    /* image defects detection,use translation mat metho */
		MV_ALG_MOVING_DETECTION,               /* moving object detection */
		MV_ALG_MONO_CAM_CALI,                  /* monocular camera calibration */
	}algType;

	/* return code */
	enum
	{
		MV_FAILED            = -4,
		MV_LIC_VALID,
		MV_MEMORY_NOT_ENOUGH,
		MV_OPEN_FILE_ERROR, 
		MV_ERROR             = 0,
		MV_OK                = 1,
		MV_PARAM_ERROR,
		MV_UNSUPPORTED
	};
	/* alg running status  */
	enum
	{
		MVALG_IDLE = 0,
		MVALG_RUNNING,
		MVALG_STOP,
	};

	typedef bool mvBool;

	typedef struct
	{
		union
		{
			int x;              /* x position */
			float xx;           /* x position */
		};
		union
		{
			int y;              /* y position */
			float yy;           /* y position */
		};
	}mvPoint;

	typedef struct
	{
		mvPoint tmp;       
		mvPoint dst;
	}mvMatchPoint;

	//typedef struct
	//{
	//	mvFPoint tmp;
	//	mvFPoint dst;
	//}mvMatchFPoint;

	typedef struct
	{
		float x;                /* x position */
		float y;                /* y position */
	}mvFPoint;

	typedef struct
	{
		mvPoint    *pnts;       /* point to points */
		int        nump;        /* num point */
		int        maxp;        /* max point */
	}mvPoints;

	typedef struct
	{
		mvPoint	  up_left;      /* up-left point */
		mvPoint   dw_right;     /* bottom-right point */
	}mvRect;

	typedef struct
	{
		mvPoint pnts[4];         /* 4-point represent of this box */
		mvPoint center;          /* center point of this box */
		float   angle;           /* angle of this box */
		float   width;           /* width */
		float   height;          /* height */
		float   area;            /* area of this box */
	}mvBox;

	typedef struct
	{
		int width;               /* width */
		int	height;              /* height */
		int nsize;               /* image size */
		int channels;            /* channel, 1: grey, 3: color */
		unsigned char *pdata;    /* point to image data */
	}mvRGBImage;

	typedef struct
	{
		int	width;               /* width */
		int	height;              /* height */
		int nsize;               /* image size */
		int channels;            /* channels */
		unsigned char *pdata;    /* point to image data */
	}mvImage;


#define MV_MAX_POLYGON   128
#define MV_MAX_PO_POINTS 32


	enum
	{
		KEY_AREA = 10,
		IM_AREA = 11,
		NOR_AREA = 12,
		NIANSHA_AREA = 20,
		MAOBIAN_AREA = 30,
		QIEGE_AREA = 40
	};

	typedef struct
	{
		int type;               /* polygon type */
		int num;                /* num polygons */
		int valid;              /* set true if this is an valid area */
		int uflag;              /* set true if need to use unite */
		int index;              /* polygon zone label index, index = 0 is use for whole image  */
		unsigned char uc;       /* lable index color (0~255) */
		unsigned int area;      /* polygon area */
		mvPoint  seed;          /* polygon seed */
		mvPoint  ppnts[MV_MAX_PO_POINTS]; /* num points */
	}mvPolygon;

	/* image input type */
	typedef enum
	{
		MV_GRAY = 0,           /* grey image type */
		MV_BGR24,              /* b g r image type */
		MV_RGB24,              /* r g b image type */
		MV_CV_IPL,             /* opencv IplImage type, gray, rgb888 mode, 1 channel, 3 channel supported */
		MV_CV_MAT,             /* opencv Mat type, dims = 2, gray, rgb888 mode, 1, channel, 3 channel supported */
		MV_YUV420_YV12,        /* YUV420P:  3-plane, Y, U, V - Y, U, V */
		MV_YUV420_YV21,        /* YUV420P:  3-plane, Y, V, U - Y, V, U */
		MV_YUV420_NV12,        /* YUV420SP: 2-plane, Y, UVUVUV - Y, UVUVUV */
		MV_YUV420_NV21         /* YUV420SP: 2-plane, Y, VUVUVU - Y, VUVUVU */
	}imageType;

	/* param methos */
	enum
	{
		MV_NO_MATCH = 0,
		MV_MATCH_SHAPE,
		MV_MATCH_FEATSPOINT,
	};

	/* image input for algs */
	typedef struct
	{
		int       index;          /* frame index */
		imageType type;           /* MV_GRAY, MV_BGR24, MV_CV_IPL, MV_YUV420_NV21 */
		int       width;          /* width */
		int       height;         /* height */
		int       depth;          /* depth of this image, default:8 */
		int       channels;       /* channel, 1: grey, 3: color */
		int       nsize;          /* size of data */
		int       wstep;          /* width step */           
		void      *pframe;        /* point to image or data */
		void      *user_param;    /* user param struct */
	}mvInputImage;

	typedef struct
	{
		mvInputImage temp;       /* template image */
		double tran_matrix[16];  /* 4x4 matrix can store the 3x3 matrix data */
		int type;                /* type of the tran_matrix, 0: 2x3, 1: 3x3, 2: 4x4 */
	}mvTempImg;

	typedef struct
	{
		mvRect   rec_bound;       /* object rect bound */
		mvPoint  center;          /* center of this object */
		mvPoints contours;        /* contours points of this object */
		int      npixels;         /* number pixels of this object */
	}tempObj;

	typedef struct
	{
		tempObj *tmp_objs;        /* template object list */
		int     num_obj;          /* num object */
	}mvTempObjs;

	typedef struct
	{
		mvRect		  rec_bound;         /* rectbound */
		mvBox         obj_box;           /* boudingbox */
		mvPoints      convex;            /* ccl convex */
		mvPoint	      pnt_center;        /* center point */
		mvPoint		  pnt_mass;          /* center mass point */
		mvPoints	  pnt_countour;      /* contour */
		unsigned int  com_label;         /* ccl lable */
		unsigned char com_color;         /* ccl color (0~255) */
		unsigned int  di_pixels;         /* di-pixel count */
		unsigned int  sum_area;          /* sumarea */
		//
		unsigned int  en_pixels;         /* enrode-pixels count */
		float         pc_pixels;         /* pixel percent */
		float         lbp_val;           /* LBP value */
		float         confidence;        /* confidence */
		mvPoint		  pnt_inside;        /* inside flag */
		mvBool		  vflag;             /* valid flag */
		unsigned int  ul_flag;
		unsigned int  sum_contour;
		unsigned int  num_contour;
		int           pair_index;        /* template index */
		float         pair_corr;         /* paircorr */
	}mvCCLItem;

	//Contour attributes data
	typedef struct
	{
		mvBool		point_valid;         /* contour points */
		mvBool		contour_valid;       /* contour sum, Contour number of pixels */
		int 		best_area;           /* best fit comp area */
	}mvCCLATTRFLAGS;

	typedef struct
	{
		int width;
		int height;
		int			    num_comp;
		int			    max_comp;
		int             num_class;   /* num classes of components */
		mvPoints	    pnts_bank;   /* point for all components */
		mvCCLATTRFLAGS  atr_flag;    /* flags */
		mvCCLItem       *ccomps;     /* connected components*/
	}mvCCLItems;

	typedef struct
	{
		double   hmat[9];           /* 3x3 persbective mat for this object */
		float    quac[8];           /* q elements */
		mvPoints contours;          /* contours points of this object */
		mvPoints convex;            /* convexhull */
		void     *shape_data;      /* shape data */
	}unMObjHeader;

	typedef struct
	{
		int      index;             /* object index */
		int      mat_index;         /* matched object index */
		int      valid;             /* valid flag */
		mvRect   rec_bound;         /* object rect bound */
		mvBox    rot_box;           /* rotate box */
		mvPoint  center;            /* center of this object */
		int      num_pixels;        /* number pixels of this object */
		float    rotation;          /* ration of this object */
		float    angle;             /* angle of this object */
		float    rel_score;         /* reliable score */
		float    scale;             /* scale factor output */
		int      is_track;          /* is track */
		int      temp_id;           /* tempId */
		int      uc_color;          /* object lable color */
		double   hmat[9];           /* 3x3 persbective mat for this object */
		float    quac[8];           /* q elements */
		mvPoints contours;          /* contours points of this object */
		mvPoints convex;            /* convexhull */
		void     *user_data;        /* shape data, cvMat, or other data*/
	}matchObj;

	typedef struct
	{
		int   num_class;            /* num Class of object */
		int   num_obj;              /* number matched object */
		int   num_valid;            /* num valid object */
		int   best_index;           /* the highest score */
		float avg_val;              /* average value */
		int   value1;               /* reserved value for future */
		int   value2;               /* reserved value for future */
		int   value3;               /* reserved value for future */
		float value4;               /* reserved value for future */
		float value5;               /* reserved value for future */
		float value6;               /* reserved value for future */
		std::vector <matchObj> mat_obj;    /* match objects */
	}mvMatchObjs;

	typedef struct
	{
		void        *user_dat;      /* usr data or struct */
		mvMatchObjs mat_objs;       /* match object list */
		mvTempObjs  tmp_objs;       /* template object list */
		mvCCLItems  *cc_items;      /* point to connected-components */
		int         reserve;
	}mvResult;


	typedef struct
	{
		void        *user_data;     /* usr data or struct */
		mvMatchObjs mat_objs;       /* match object list */
		mvTempObjs  tmp_objs;       /* template object list */
		mvCCLItems  *ccls;          /* point to ccl - components */
		int         reserve;
	}mvResult2;

	typedef struct
	{
		int       num_poly;                 /* num polygon */
		unsigned int area;                  /* output, total area */
		mvImage   *roi_map;                 /* roi image, set to null if don't want to use it */
		mvPolygon polys[MV_MAX_POLYGON];    /* roi detect area */
	}mvDetRoi;

	typedef struct
	{
		void *user_param;          /* undefined parameters input */
		int org_width;             /* original image width */
		int org_height;            /* original image height */
		int channels;              /* original image channel */
		int scl_width;             /* scale image width */
		int scl_height;            /* scale image width */
		int loc_width;             /* locate image width */
		int loc_height;            /* locate image width */
		mvPoint loc_offset;        /* locate offset relate to the original image */
		float x_scale;             /* x pos scale factor */
		float y_scale;             /* y pos scale factor */
		mvRect loc_roi;            /* roi area of alg */
		int loc_flag;              /* if set the locate image flag */
		mvDetRoi det_roi;          /* det roi area on the original image */
		mvDetRoi map_roi;          /* map of the det ro area */
		char cfg_path[512];        /* alg config file path */
	}mvRunCfg;

	typedef struct
	{
		void *user_param;          /* undefined parameters input */
		/* const parameters */
		int   sub_metho;           /* algrithm metho: 0, 1, 2, 3... */
		float scale;               /* default to 1.0 */
		int   disp_level;          /* display level */
		int   user_defi;           /* userDef1*/
		int   user_def2i;          /* userDef1*/
		int   user_def3i;          /* userDef1*/
		float user_def4f;          /* userDef1*/
		float user_def5f;          /* userDef1*/
		float user_def6f;          /* userDef1*/
		float user_def7f;          /* userDef1*/

		/* user-defined parameters */
		int   diff_val;            /* differrence value default:50  */
		int   min_area;            /* min area */
		int   max_area;            /* max area */
		int   min_pixels;          /* minimal object pixel size */
		int   max_pixels;          /* maximal object pixel size */
		int   min_gray;            /* minimal gray value */
		int   max_gray;            /* max gray value */
		float min_wh_rate;         /* min(w,h)/max(w,h) */
		float max_wh_rate;         /* min(w,h)/max(w,h) */
		float ppa;                 /* edge-pixels/area  */
		float gr_thres;            /* gray difference threshold */
		float co_thres;            /* co threshold */
		float con_likeness;        /* contours likeness */
		float gr_likeness;         /* gray-level linkness */
		float conf_thres;          /* confidenceThreshold, default 0.6*/
		int   edge_thres;          /* edge maxThres */
		int   edge_flag;           /* remove edge flag */
		int   use_filter;          /* use filter for match object*/
		int   use_clsfier;         /* use classifier £¨svm/adaboost/ssd-deep cnn module)*/
		int   use_mapper;          /* use mapper */
		int   use_deepl;           /* use deep-learning model*/
		int   reserver5;           /* resever */
		int   reserver6;           /* resever */
		int   reserver7;           /* resever */

		float reserver8;           /* resever */
		float reserver9;           /* resever */
		float reserver10;          /* resever */
		float reserver11;          /* resever */
		float reserver12;          /* resever */
		/* user-defined parameters */
	}mvEngineCfg;

	typedef struct
	{
		void    *user_param;        /* pass the user_data struct */
		int     online_temp;        /* set to true if use this init param */
		char    cfg_path[255];      /* mv-config path */
		mvInputImage tmp_img;       /* tmplate object frame input */
		mvRect  tmp_roi;            /* loc-roi */
		mvPoint tmp_offset;         /* offset in the original image [width, heigth] */
		int     reserved;
	}initParam;

	typedef struct
	{
		algType type;               /* alg process type */
		int sub_type;               /* subset type of the alg type */
		int init_flag;              /* engine start flag, set false will be restart */
		int run_status;             /* running status */
		mvRGBImage  cur_img;        /* current processed image */
		mvImage     com_img;        /* 1 channel, connect components on this image */
		mvImage		edge_img;       /* 1 channel, contours edge on this image */
		mvImage		bin_img;        /* 1 channel, binary image */
		mvRunCfg    run_cfg;        /* running config parameters */
		mvResult    result;         /* results of this alg */
		mvEngineCfg alg_params;     /* config parameters */
	}algDllHandle;

	typedef struct
	{
		mvPoint      h1;
		mvFPoint     h2;
		mvPoints     h3;
		mvRect       h4;
		mvBox        h5;
		mvRGBImage   h6;
		mvImage      h7;
		mvPolygon    h8;
		mvInputImage h9;
		tempObj      h10;
		mvTempObjs   h11;
		matchObj     h13;
		mvMatchObjs  h14;
		mvResult     h15;
		mvDetRoi     h16;
		mvRunCfg     h17;
		initParam    h18;
		algDllHandle h19;
	}mvVersionHeader;

	typedef struct
	{
		int   num_poly;                        /* num polygon */
		int   uc_lable[MV_MAX_POLYGON];
		int   res1[MV_MAX_POLYGON];
		float res2[MV_MAX_POLYGON];
		float res3[MV_MAX_POLYGON];
	}mvDetResult;

#define MV_VERSION_HEADER  sizeof(mvVersionHeader)

#ifdef __cplusplus
}
#endif

/*

mv X-Vison SDK check License

return 0, no license - dog
return 1, license ok
return 2, license valid

else undefine.

*/
extern int mvCheckXVisonLicense(void);

#endif


/* algorithm display-Level discription */

/*
disp_level(0x01<<0)  ---------- obj-valid
disp_level(0x01<<1)  ---------- algRoi
disp_level(0x01<<2)  ---------- none
disp_level(0x01<<3)  ---------- obj-rectangle
disp_level(0x01<<4)  ---------- obj-roatebox
disp_level(0x01<<5)  ---------- "+"
disp_level(0x01<<6)  ---------- none
disp_level(0x01<<7)  ---------- obj-contours
disp_level(0x01<<8)  ---------- obj-centerPos
disp_level(0x01<<9)  ---------- obj-rel score
disp_level(0x01<<10) ---------- obj-ang
disp_level(0x01<<11) ---------- template-contours
disp_level(0x01<<12) ---------- tracking-line
disp_level(0x01<<13) ---------- binaray image mode
disp_level(0x01<<14) ---------- blobs image mode
disp_level(0x01<<15) ---------- detRoi-map
disp_level(0x01<<16) ---------- draw thickness(bit 0)
disp_level(0x01<<17) ---------- draw thickness(bit 1)
disp_level(0x01<<18) ---------- draw thickness(bit 2)
disp_level(0x01<<19) ---------- draw thickness(bit 3)
disp_level(0x01<<20) ---------- draw label & location color
disp_level(0x01<<21) ---------- draw gray historgram
disp_level(0x01<<22) ---------- reserved
disp_level(0x01<<23) ---------- reserved
disp_level(0x01<<24) ---------- reserved
disp_level(0x01<<25) ---------- reserved
disp_level(0x01<<26) ---------- reserved
*/

