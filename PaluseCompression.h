#ifndef _PALUSECOMPRESSION_H_
#define _PALUSECOMPRESSION_H_
#include <cuda_runtime.h>
#include <cufft.h>
#include <cuComplex.h>
#include <cstdio>
#define Pi 3.14159265358

typedef struct ProcessPara {
    unsigned int nrn;
    unsigned int nan;
	unsigned int L;
	unsigned int M;
	double Lambda;
	double B;
	double prf;
	double fs;
	double Rs;
	double K;
    double Tp;
	
}ProcessPara_t;

void paluseCompression(float* datain_real,float* datain_img,float* dataout_real,float* dataout_img ,ProcessPara_t Para);
unsigned int nextPowerOfTwo(unsigned int n);


#endif
