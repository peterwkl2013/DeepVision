/*************************************************************************
*
X-Vision SDK                                          *
*
File Discription:                                                        *
mv image algorithm Dll export functions                              *
discription:                                                             *
machine vision, object location, pattern recognition,                *
segmentation etc process                                             *
Author:                                                                  *
X-Vision RD team                                                     *
WSN@2014/09/14                                                       *
**************************************************************************/

#ifndef  __DLLMVINTERFACE_H__
#define  __DLLMVINTERFACE_H__

#include "mvSDK_type.h"
#include "mvSDK_interface.h"

#ifndef MV_RELEASE
#pragma comment(lib,"LibDeepVision_x64.lib")
#else
#pragma comment(lib,"LibDeepVision_x64.lib")
#endif

using namespace std;

#ifdef __cplusplus
extern "C" {
#endif

    /* Mv image algorithmn Lib Functions */
    int      FDLLMvLibLoad(void);
    void     FDLLMvLibFree(void);

    /* check mv version
    @ input: sizeof(mvVersionHeader)
    */
    int     FDLLMvCheckVersion(int nsize);
    int     mvCheckVersion(int nsize);
    /* alloc an instance of alg handle
    @ input: width,  image width
    @ input: height, image height
    @ input: type, alg process type
    @ input: para, parameters of alg process
    return ppAlg hanlde
    */
    void*    FDLLMvInstanceAlloc(int width, int height, algType type, initParam *para);


    /* alloc an instance of alg handle
    @ input: ppalg, alg handle
    */
    void     FDLLMvInstanceDelete(void *ppAlg);
    void     mvInstanceDelete(void *ppAlg);

    /* alg process funtion
    @ input: ppalg, alg handle
    @ input: imageInput, image
    return mv return code
    */
    int      FDLLMvAlgProcess(void *ppAlg, mvInputImage *imageInput);
    int      mvAlgProcess(void *ppAlg, mvInputImage *imageInput);

    /* setting the scale factor of the alg process
    @ input: cfgpath, the configure path, or NULL
    @ input: scl, the scl, eg: 0.5, 1, 2
    return mv return code
    */
    int     FDLLMvSetAlgScale(char *cfgpath, float scl);
    int     mvSetAlgScale(char *cfgpath, float scl);

    /* read the configure file
    @ output: param, the parameters
    @ input:  nsize, nsize = sizeof(param)
    @ input: cfgpath, cfgpath, the configure path, or NULL
    return mv return code
    */
    int     FDLLMvReadCfgFile(mvEngineCfg *param, int nsize, char *cfgpath);
    int     mvReadCfgFile(mvEngineCfg *param, int nsize, char *cfgpath);
    /* setting parameters
    @ input: ppalg, alg handle
    @ input: param, the parameters
    @ input:  nsize, nsize = sizeof(param)
    @ input: cfgpath, cfgpath, the configure path, or NULL
    return mv return code
    */
    int     FDLLMvParamSetting(void *ppAlg, mvEngineCfg *param, int nszie, char *cfgpath);
    int     mvParamSetting(void *ppAlg, mvEngineCfg *param, int nszie, char *cfgpath);
    /* setting user parameters
    @ input: ppalg, alg handle
    @ input: param, the parameters
    return mv return code
    */
    int     FDLLMvUserParamSetting(void *ppAlg, void *param);
    int     mvUserParamSetting(void *ppAlg, void *param);
    /* update parameters
    @ input: ppalg, alg handle
    @ input: param, the parameters
    return mv return code
    */
    int     FDLLMvParamUpdate(void *ppAlg, mvEngineCfg *param, int nsize);
    int     mvParamUpdate(void *ppAlg, mvEngineCfg *param, int nsize);

    /* setting detect roi area
    @ input: ppalg, alg handle
    @ input: para, parameters of alg process
    */


    int     FDLLMvSetDetRoiArea(void *ppAlg, mvDetRoi roi, char *cfgpath);


    /* get map point
    @ input: ppalg, alg handle
    @ input: x, x position
    @ input: y, y position
    @ input: type,  map point from template or input image
    return mv return code
    */
    mvFPoint FDLLMvGetMapPoint(void *ppAlg, float x, float y, int type);
    mvFPoint mvGetMapPoint(void *ppAlg, float x, float y, int type);

    /* setting detect roi area
    @ input: ppalg, alg handle
    @ input: res, return the process results of the alg
    no return
    */
    void    FDLLMvResultsDrawAndDisplay(void *ppAlg, mvResult *res);
    void    mvMatchObjsDrawAndDisplay(void *ppAlg, mvResult *res);

    /* mv qr decode alg.
    @ input: imageInput, image
    @ return: result (symObj struct)
    */
    int FDLLMvScannerProcessWrap(mvInputImage *imageInput, void *res);
    int mvScannerProcessWrap(mvInputImage *imageInput, void *res);
#ifdef __cplusplus
}
#endif



#endif


/* algorithm display-Level discription */

/*
disLevel(0x01<<0)  ---------- obj-valid
disLevel(0x01<<1)  ---------- algRoi
disLevel(0x01<<2)  ---------- none
disLevel(0x01<<3)  ---------- obj-rectangle
disLevel(0x01<<4)  ---------- obj-roatebox
disLevel(0x01<<5)  ---------- "+"
disLevel(0x01<<6)  ---------- none
disLevel(0x01<<7)  ---------- obj-contours
disLevel(0x01<<8)  ---------- obj-centerPos
disLevel(0x01<<9)  ---------- obj-rel score
disLevel(0x01<<10)  --------- obj-ang
disLevel(0x01<<11) ---------- template-contours
disLevel(0x01<<12) ---------- tracking-line
disLevel(0x01<<13) ---------- binaray image mode
disLevel(0x01<<14) ---------- blobs image mode
disLevel(0x01<<15) ---------- detRoi-map
disLevel(0x01<<16) ---------- draw thickness(bit 0)
disLevel(0x01<<17) ---------- draw thickness(bit 1)
disLevel(0x01<<18) ---------- draw thickness(bit 2)
disLevel(0x01<<19) ---------- draw thickness(bit 3)
disLevel(0x01<<20) ---------- draw label & location color
disLevel(0x01<<21) ---------- draw gray historgram
disLevel(0x01<<22) ---------- reserved
disLevel(0x01<<23) ---------- reserved
disLevel(0x01<<24) ---------- reserved
disLevel(0x01<<25) ---------- reserved
disLevel(0x01<<26) ---------- reserved
*/
