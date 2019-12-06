#include "DllMVInterface.h"
/*****************************************************

File Discription: mv image algorithm Dll export functions
discription: object location, pattern recognition process

Author: WSN@2014/09/14
******************************************************/

#include <stdio.h>

#define MV_USE_MSVC_API
//#define MV_USE_QT_API
//#define MV_USE_LINUX_API

#if defined (MV_USE_MSVC_API)
#include <Windows.h>
static HINSTANCE mvSDKHandle = NULL;
#ifdef MV_RELEASE
#ifdef MV_WIN32
//#define LIBMV_PATH "./X-Vision4Win.dll"
#define LIBMV_PATH "./LibX-Vision.dll"
#elif MV_WIN64
//#define LIBMV_PATH "./X-Vision4Win64.dll"
#define LIBMV_PATH "./LibX-Vision_x64.dll"
#elif MV_ARM_32
#define LIBMV_PATH "./libX-Vision4ARM.dll"
#elif MV_ARM_64
#define LIBMV_PATH "./libX-Vision4ARM64.dll"
#endif
#else
#ifdef MV_WIN32
//#define LIBMV_PATH "./X-Vision4Win.dll"
#define LIBMV_PATH "./LibX-Vision_X86d.dll"
#elif MV_WIN64
#define LIBMV_PATH "./LibX-Vision_x64d.dll"
#elif MV_ARM_32
#define LIBMV_PATH "./libX-Vision4ARM.dll"
#elif MV_ARM_64
#define LIBMV_PATH "./libX-Vision4ARM64.dll"
#endif
#endif

#elif defined (MV_USE_QT_API)
#include <QLibrary>
static QLibrary mvlib;
#define LIBMV_PATH "iMV4Android.so"
#elif defined (MV_USE_LINUX_API)
#include <dlfcn.h>
static void *mvSDKHandle = NULL;
#define LIBMV_PATH "iMV4Linux.so"
#endif

#include "DllMVInterface.h"


static int refcnt = 0;

typedef int       (*DLLMvCheckVersion)(int nsize);
typedef	void*     (*DLLMvInstanceAlloc)(int width, int height, algType type, initParam *para);
typedef	void*     (*DLLMvInstanceDelete)(void *ppAlg);
typedef int       (*DLLMvAlgProcess)(void *ppAlg, mvInputImage *imageInput);
typedef	mvDetRoi  (*DLLMvGetMapDetRoi)(void *ppAlg);
typedef	mvFPoint  (*DLLMvGetMapPoint)(void *ppAlg, float x, float y, int type);
typedef int       (*DLLFDLLMvSetAlgScale)(char *cfgpath, float scl);
typedef	int       (*DLLMvParamUpdate)(void *ppAlg, mvEngineCfg *param, int nsize);
typedef void      (*DLLMvAlgTempReset)(void *ppAlg, initParam *para);
typedef	int       (*DLLMvParamSetRoi)(void *ppAlg, mvRect roi);
typedef	int       (*DLLMvSetDetRoiArea)(void *ppAlg, mvDetRoi roi, char *cfgpath);
typedef int       (*DLLMvParamSetting)(void *ppAlg, mvEngineCfg *param, int nszie, char *cfgpath);
typedef int       (*DLLMvUserParamSetting)(void *ppAlg, void *param);
typedef int       (*DLLMvReadCfgFile)(mvEngineCfg *param, int nsize, char *cfgpath);
typedef void      (*DLLMvResultsDrawAndDisplay)(void *ppAlg, mvResult *res);
typedef void      (*DLLMvMatchNotify)(void *ppAlg);
typedef int       (*DLLMvLibRegister)(char *licNum);
typedef char*     (*DLLMvGetLog)(void);
typedef void      (*DLLMvLogClean)(void);



typedef int (*DLLMvScannerProcessWrap)(mvInputImage *imageInput, void *res);

//
#ifdef SAIYI
//int FDLLMvResultsCheckFace(char *pfilePath,int checkType,outResults *outRe);
typedef int (*DLLMvResultsCheckFace)(void *ppAlg, char *pfilePath,int checkType,outResults *outRe);
typedef int (*DLLMvResultsCheckGrayAndEdge)(void *ppAlg, char *pfilePath,int checkType,outResults *outRe);
typedef int (*DLLMvResultsCheckRed)(void *ppAlg, char *pfilePath,int checkType,mvResult *res);
typedef void* (*DLLMvInstanceAlloc2)(char *pfilePath, char *cfgpath);
#endif

static DLLMvCheckVersion             _tMvCheckVersion     = NULL;
static DLLMvInstanceAlloc            _tMvInstanceAlloc    = NULL;
static DLLMvInstanceDelete           _tMvInstanceDelete   = NULL;
static DLLMvAlgProcess               _tMvAlgProcess       = NULL;
static DLLMvGetMapDetRoi             _tMvGetMapDetRoi     = NULL;
static DLLMvGetMapPoint              _tMvGetMapPoint      = NULL;
static DLLFDLLMvSetAlgScale          _tMvSetAlgScale      = NULL;
static DLLMvParamUpdate              _tMvParamUpdate      = NULL;
static DLLMvAlgTempReset             _tMvAlgTempReset     = NULL;
static DLLMvParamSetRoi              _tMvParamSetRoi      = NULL;
static DLLMvSetDetRoiArea            _tMvSetDetRoiArea    = NULL;
static DLLMvParamSetting             _tMvParamSetting     = NULL;
static DLLMvUserParamSetting         _tMvUserParamSetting = NULL;
static DLLMvReadCfgFile              _tMvReadCfgFile      = NULL;
static DLLMvResultsDrawAndDisplay    _tMvResultsDrawAndDisplay = NULL;
static DLLMvMatchNotify              _tMvMatchNotify           = NULL;
static DLLMvLibRegister              _tMvLibRegister           = NULL;
static DLLMvGetLog                   _tMvGetLog                = NULL;
static DLLMvLogClean                 _tMvLogClean              = NULL;

static DLLMvScannerProcessWrap       _tMvScannerProcessWrap = NULL;


int FDLLMvCheckVersion(int nsize)
{
	return _tMvCheckVersion(nsize);
}

void* FDLLMvInstanceAlloc(int width, int height, algType type, initParam *para)
{
    return(void*)_tMvInstanceAlloc(width, height, type, para);
}

void FDLLMvInstanceDelete(void *ppAlg)
{
    _tMvInstanceDelete(ppAlg);
}

int FDLLMvAlgProcess(void *ppAlg, mvInputImage *imageInput)
{
    return (_tMvAlgProcess(ppAlg, imageInput));
}

mvDetRoi FDLLMvGetMapDetRoi(void *ppAlg)
{
    return (_tMvGetMapDetRoi(ppAlg));
}

mvFPoint FDLLMvGetMapPoint(void *ppAlg, float x, float y, int type)
{
    return (_tMvGetMapPoint(ppAlg, x, y, type));
}

int FDLLMvSetAlgScale(char *cfgpath, float scl)
{
	return (_tMvSetAlgScale(cfgpath, scl));
}

int FDLLMvParamUpdate(void *ppAlg, mvEngineCfg *param, int nsize)
{
    return (_tMvParamUpdate(ppAlg, param, nsize));
}

void FDLLmvAlgTempReset(void *ppAlg, initParam *para)
{
    return (_tMvAlgTempReset(ppAlg, para));
}

int FDLLMvParamSetRoi(void *ppAlg, mvRect roi)
{
    return (_tMvParamSetRoi(ppAlg, roi));
}

int FDLLMvSetDetRoiArea(void *ppAlg, mvDetRoi roi, char *cfgpath)
{
    return (_tMvSetDetRoiArea(ppAlg, roi, cfgpath));
}

int FDLLMvParamSetting(void *ppAlg, mvEngineCfg *param, int nszie, char *cfgpath)
{
    return (_tMvParamSetting(ppAlg,param, nszie, cfgpath));
}

int FDLLMvUserParamSetting(void *ppAlg, void *param)
{
	return (_tMvUserParamSetting(ppAlg, param));
}

int FDLLMvReadCfgFile(mvEngineCfg *param, int nsize, char *cfgpath)
{
	return (_tMvReadCfgFile(param, nsize, cfgpath));
}

void FDLLMvResultsDrawAndDisplay(void *ppAlg, mvResult *res)
{
    return (_tMvResultsDrawAndDisplay(ppAlg, res));
}

void FDLLMvMatchNotify(void *ppAlg)
{
    return (_tMvMatchNotify(ppAlg));
}

int FDLLMvLibRegister(char *licNum)
{
    return (_tMvLibRegister(licNum));
}

char* FDLLMvGetLog(void)
{
    return (_tMvGetLog());

}

void FDLLMvLogClean(void)
{
    return (_tMvLogClean());
}

int FDLLMvScannerProcessWrap(mvInputImage *imageInput, void *res)
{
	return (_tMvScannerProcessWrap(imageInput, res));
}


int FDLLMvLibLoad(void)
{
    if( refcnt == 0)
    {
#if defined (MV_USE_MSVC_API)
        mvSDKHandle = ::LoadLibrary(LIBMV_PATH);
        if (mvSDKHandle == 0)
        {
            printf("Load iMV DLL error\n");
            return MV_ERROR;
        }
		//////

		/////

		_tMvCheckVersion    = (DLLMvCheckVersion)::GetProcAddress(mvSDKHandle, "mvCheckVersion");
        _tMvInstanceAlloc   = (DLLMvInstanceAlloc)::GetProcAddress(mvSDKHandle, "mvInstanceAlloc");
        _tMvInstanceDelete  = (DLLMvInstanceDelete)::GetProcAddress(mvSDKHandle, "mvInstanceDelete");
        _tMvAlgProcess      = (DLLMvAlgProcess)::GetProcAddress(mvSDKHandle, "mvAlgProcess");
		_tMvGetMapDetRoi    = (DLLMvGetMapDetRoi)::GetProcAddress(mvSDKHandle, "mvGetMapDetRoi");
		_tMvGetMapPoint  = (DLLMvGetMapPoint)::GetProcAddress(mvSDKHandle, "mvGetMapPoint");
		_tMvSetAlgScale  = (DLLFDLLMvSetAlgScale)::GetProcAddress(mvSDKHandle, "mvSetAlgScale");
        _tMvParamUpdate   = (DLLMvParamUpdate)::GetProcAddress(mvSDKHandle, "mvParamUpdate");
		//_tMvAlgTempReset    = (DLLMvAlgTempReset)::GetProcAddress(mvSDKHandle, "mvAlgTempReset");
        //_tMvParamSetRoi    = (DLLMvParamSetRoi)::GetProcAddress(mvSDKHandle, "mvParamSetRoi");
        //_tMvSetDetRoiArea   = (DLLMvSetDetRoiArea)::GetProcAddress(mvSDKHandle, "mvSetDetRoiArea");
        _tMvParamSetting    = (DLLMvParamSetting)::GetProcAddress(mvSDKHandle, "mvParamSetting");
		_tMvUserParamSetting = (DLLMvUserParamSetting)::GetProcAddress(mvSDKHandle, "mvUserParamSetting");
		_tMvReadCfgFile     = (DLLMvReadCfgFile)::GetProcAddress(mvSDKHandle, "mvReadCfgFile");
		_tMvReadCfgFile     = (DLLMvReadCfgFile)::GetProcAddress(mvSDKHandle, "mvReadCfgFile");
        _tMvResultsDrawAndDisplay  = (DLLMvResultsDrawAndDisplay)::GetProcAddress(mvSDKHandle, "mvMatchObjsDrawAndDisplay");
        //_tMvMatchNotify     = (DLLMvMatchNotify)::GetProcAddress(mvSDKHandle, "mvMatchNotify");
        //_tMvLibRegister     = (DLLMvLibRegister)::GetProcAddress(mvSDKHandle, "mvLibRegister");
        //_tMvGetLog          = (DLLMvGetLog)::GetProcAddress(mvSDKHandle, "mvGetLog");
        //_tMvLogClean        = (DLLMvLogClean)::GetProcAddress(mvSDKHandle, "mvLogClean");

		_tMvScannerProcessWrap = (DLLMvScannerProcessWrap)::GetProcAddress(mvSDKHandle, "mvScannerProcessWrap");

#elif defined (MV_USE_QT_API)
         QLibrary mvlib(LIBMV_PATH);
         if (!mvlib.load())
         {
             printf("Load iMV DLL error\n");
             return MV_ERROR;
         }

		_tMvCheckVersion    = (DLLMvCheckVersion)mvlib.resolve("mvCheckVersion");
        _tMvInstanceAlloc   = (DLLMvInstanceAlloc)mvlib.resolve("mvInstanceAlloc");
        _tMvInstanceDelete  = (DLLMvInstanceDelete)mvlib.resolve("mvInstanceDelete");

        _tMvAlgProcess    = (DLLMvMatchProcess)mvlib.resolve("mvAlgProcess");
		_tMvGetMapDetRoi    = (DLLMvGetMapDetRoi)mvlib.resolve("mvGetMapDetRoi");
		_tMvGetMapPoint  = (DLLMvGetMapPoint)mvlib.resolve("mvGetMapPoint");
		_tMvSetAlgScale  = (DLLFDLLMvSetAlgScale)mvlib.resolve("FDLLMvSetAlgScale");
        _tMvParamUpdate     = (DLLMvParamUpdate)mvlib.resolve("mvParamUpdate");
		_tMvAlgTempReset    = (DLLMvAlgTempReset)mvlib.resolve("mvAlgTempReset");
        _tMvParamSetRoi     = (DLLMvParamSetRoi)mvlib.resolve("mvParamSetRoi");
        _tMvSetDetRoiArea   = (DLLMvSetDetRoiArea)mvlib.resolve("mvSetDetRoiArea");
        _tMvParamSetting    = (DLLMvParamSetting)mvlib.resolve("mvParamSetting");
		_tMvUserParamSetting = (DLLMvUserParamSetting)mvlib.resolve("mvUserParamSetting");
		_tMvReadCfgFile     = (DLLMvReadCfgFile)mvlib.resolve("mvReadCfgFile");
		_tMvReadCfgFile     = (DLLMvReadCfgFile)mvlib.resolve("mvReadCfgFile");
        _tMvResultsDrawAndDisplay  = (DLLMvResultsDrawAndDisplay)mvlib.resolve("mvMatchObjsDrawAndDisplay");
        _tMvMatchNotify     = (DLLMvMatchNotify)mvlib.resolve("mvMatchNotify");
        _tMvLibRegister     = (DLLMvLibRegister)mvlib.resolve("mvLibRegister");
        _tMvGetLog          = (DLLMvGetLog)mvlib.resolve("mvGetLog");
        _tMvLogClean        = (DLLMvLogClean)mvlib.resolve("mvLogClean");

#elif defined (MV_USE_LINUX_API)
        mvSDKHandle = dlopen(LIBMV_PATH, RTLD_NOW);
        if (mvSDKHandle == NULL)
        {
            printf("Load iMV DLL error\n");
            return MV_ERROR;
        }

		_tMvCheckVersion    = (DLLMvCheckVersion)dlsym("mvCheckVersion");
        _tMvInstanceAlloc   = (DLLMvInstanceAlloc)dlsym(mvSDKHandle, "mvInstanceAlloc");
        _tMvInstanceDelete  = (DLLMvInstanceDelete)dlsym(mvSDKHandle, "mvInstanceDelete");
        _tMvAlgProcess    = (DLLMvMatchProcess)dlsym(mvSDKHandle, "mvAlgProcess");
		_tMvGetMapDetRoi    = (DLLMvGetMapDetRoi)dlsym(mvSDKHandle,"mvGetMapDetRoi");
		_tMvGetMapPoint  = (DLLMvGetMapPoint)dlsym(mvSDKHandle, "mvGetMapPoint");
        _tMvParamUpdate     = (DLLMvParamUpdate)dlsym(mvSDKHandle, "mvParamUpdate");
		_tMvAlgTempReset    = (DLLMvAlgTempReset)dlsym(mvSDKHandle, "mvAlgTempReset");
        _tMvParamSetRoi     = (DLLMvParamSetRoi)dlsym(mvSDKHandle, "mvParamSetRoi");
        _tMvSetDetRoiArea   = (DLLMvSetDetRoiArea)dlsym(mvSDKHandle,"mvSetDetRoiArea");
        _tMvParamSetting    = (DLLMvParamSetting)dlsym(mvSDKHandle, "mvParamSetting");
		_tMvUserParamSetting = (DLLMvUserParamSetting)dlsym(mvSDKHandle,"mvUserParamSetting");
		_tMvReadCfgFile     = (DLLMvReadCfgFile)dlsym("mvReadCfgFile");
		_tMvReadCfgFile     = (DLLMvReadCfgFile)dlsym("mvReadCfgFile");
        _tMvResultsDrawAndDisplay  = (DLLMvResultsDrawAndDisplay)dlsym(mvSDKHandle, "mvMatchObjsDrawAndDisplay");
        _tMvMatchNotify     = (DLLMvMatchNotify)dlsym(mvSDKHandle, "mvMatchNotify");
        _tMvLibRegister     = (DLLMvLibRegister)dlsym(mvSDKHandle, "mvLibRegister");
        _tMvGetLog          = (DLLMvGetLog)dlsym(mvSDKHandle, "mvGetLog");
        _tMvLogClean        = (DLLMvLogClean)dlsym("mvLogClean");

#endif
        if (!_tMvCheckVersion ||!_tMvInstanceAlloc ||!_tMvInstanceDelete
            ||!_tMvAlgProcess /*|| !_tMvGetMapPoint*/ /*|| !_tMvGetMapDetRoi*/||!_tMvParamUpdate
			/*||!_tMvAlgTempReset*/
            ||/*!_tMvSetDetRoiArea ||*/!_tMvParamSetting ||!_tMvUserParamSetting || ! _tMvReadCfgFile
            ||!_tMvResultsDrawAndDisplay /*||!_tMvMatchNotify ||!_tMvLibRegister
            ||!_tMvGetLog ||!_tMvLogClean*//* || !_tMvScannerProcessWrap*/)
        {
            printf("Load Mv DLL error!\n");
            return MV_ERROR;
        }

        refcnt++;

        return MV_OK;
    }

    refcnt++;
    return MV_OK;
}

void FDLLMvLibFree(void)
{
    if (refcnt)
        refcnt--;

    if (refcnt == 0)
    {
#ifdef SAIYI
		_tDLLMvInstanceAlloc2  = NULL;
		_tMvResultsCheckFace   = NULL;
		_tDLLMvResultsCheckRed = NULL;
#endif
        _tMvInstanceAlloc   = NULL;
        _tMvInstanceDelete  = NULL;
        _tMvAlgProcess      = NULL;
		_tMvGetMapPoint     = NULL;
		_tMvSetAlgScale     = NULL;
        _tMvParamUpdate     = NULL;
		_tMvAlgTempReset    = NULL;
        _tMvParamSetRoi     = NULL;
        _tMvSetDetRoiArea   = NULL;
        _tMvParamSetting    = NULL;
		_tMvUserParamSetting = NULL;
		_tMvReadCfgFile     = NULL;
        _tMvResultsDrawAndDisplay = NULL;
        _tMvMatchNotify     = NULL;
        _tMvLibRegister     = NULL;
        _tMvGetLog          = NULL;
        _tMvLogClean        = NULL;

#if defined (MV_USE_MSVC_API)
        //free library
        ::FreeLibrary(mvSDKHandle);
#elif defined (MV_USE_QT_API)
        //free library
#elif defined (MV_USE_LINUX_API)
        //free library
        dlclose(mvSDKHandle);
#endif
    }
}


