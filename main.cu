#include <matio.h>
#include <iostream>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <cuda_runtime.h>
#include "PaluseCompression.h"

using namespace std;    


int main(int argc,char** argv)
{
    const char* filename;
    char outputfilename[100];
    mat_t* file = nullptr;
    ProcessPara_t Para;
    float* dataoutReal = nullptr;
    float* dataoutImg = nullptr;
 
    if (argc > 2) {
        std::cerr << "指定文件:" << argv[0] << " [mat文件路径]\n";
        return 1;
    }
   else if(argc == 2){
        filename = argv[1];
        file = Mat_Open(filename, MAT_ACC_RDONLY);
   }
   else{
        filename = "data.mat";
        file = Mat_Open(filename, MAT_ACC_RDONLY);
    }
    if (!file) {
        std::cerr << "无法打开 " << filename << "  该文件或不存在\n";
        return 1;
    }  
    matvar_t* echoData = Mat_VarRead(file, "echoData");
    if (!echoData) {
        std::cerr << "读取变量失败\n";
        Mat_Close(file);
        return 1;
    }
     matvar_t* Fs = Mat_VarRead(file, "fs");
     Para.fs = *(double*)Fs->data;
     matvar_t* B = Mat_VarRead(file, "B");
     Para.B = *(double*)B->data;
     matvar_t* nan = Mat_VarRead(file, "nan");
     Para.nan = *(double*)nan->data;
     matvar_t* nrn = Mat_VarRead(file, "nrn");
     Para.nrn = *(double*)nrn->data;          
     matvar_t* TP = Mat_VarRead(file, "Tp");
     Para.Tp = *(double*)TP->data;
     Para.K = Para.B /Para.Tp;
     Para.L = Para.fs * Para.Tp;
     Para.M = nextPowerOfTwo(round(Para.L + Para.nrn -1));
    Mat_VarFree(Fs);
    Mat_VarFree(B);
    Mat_VarFree(nan);
    Mat_VarFree(nrn);
    Mat_VarFree(TP);
    Mat_Close(file);
    // 判断是否为复数矩阵
    if (!echoData->isComplex)
    {
        printf("该变量不是复数矩阵\n");
        Mat_VarFree(echoData);
    }
    if(echoData->dims[0] != Para.nrn || echoData->dims[1] != Para.nan)
    {
        printf("行或列参数不匹配\n");
    }
    cudaMallocHost(&dataoutReal, Para.nrn * Para.nan * sizeof(float));
    cudaMallocHost(&dataoutImg, Para.nrn * Para.nan * sizeof(float));
    mat_complex_split_t* complexData = static_cast<mat_complex_split_t*>(echoData->data);
    paluseCompression(static_cast<float*>(complexData->Re),static_cast<float*>(complexData->Im),dataoutReal,dataoutImg ,Para);
    size_t len = strlen(filename);
    if (len >= 4) {
        snprintf(outputfilename, sizeof(outputfilename),"%.*s_FFT_Result.mat", (int)(len - 4), filename);
    }
    file = Mat_CreateVer(outputfilename,nullptr,MAT_FT_MAT5);
    if (file == nullptr)
    {
      printf("MAT file create failed\n");
      return false;
    }
    size_t dims[2] = {Para.nrn, Para.nan};
    complexData->Re = dataoutReal;
    complexData->Im = dataoutImg;
    matvar_t* matVar = Mat_VarCreate("fft_result",MAT_C_SINGLE,MAT_T_SINGLE,2,dims,(void*)complexData,MAT_F_COMPLEX);
    int status = Mat_VarWrite(file,matVar,MAT_COMPRESSION_NONE);
    if (status != 0)
    {
        printf("Mat_VarWrite failed\n");
    }
    Mat_VarFree(echoData);
   // Mat_VarFree(echoData);
    Mat_VarFree(matVar);
    Mat_Close(file);
    
    if(!dataoutReal || !dataoutImg){
        cudaFreeHost(dataoutReal);
        cudaFreeHost(dataoutImg);
    }
    return 0;
}//main end