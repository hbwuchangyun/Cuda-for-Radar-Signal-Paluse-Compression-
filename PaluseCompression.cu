#include "PaluseCompression.h"

/* M为距离向点数，N为脉冲数量                           */
__global__  void multiplyFilter(const cuFloatComplex* datain,cuFloatComplex* dataout, const cuFloatComplex* filter,int M,int N)
{
    __shared__ float  scale;
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= M*N ) return;
    int row = (int) (i%M);
    scale = 1.0f/M;
    cuFloatComplex a = filter[row];
    a = make_cuFloatComplex(a.x*scale,-a.y*scale);
    cuFloatComplex b =  datain[i];
    dataout[i] = cuCmulf(a,b);
}
__global__ void matchFilterInint(ProcessPara_t para,cuFloatComplex* filter,int M,int L)
{
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if( i >= M) return;
    cuFloatComplex zero  = make_cuFloatComplex(0.0f,0.0f);
    if(i < L){
        double t = i / para.fs - para.Tp / 2;
        double phase = Pi * para.K * t * t;
        filter[i] = make_cuFloatComplex(cos(phase),sin(phase));
    }
    else{
        filter[i] = zero;
    }    
}
//将实部和虚部数据转换为复数数据，nrn为距离向点数，nan为脉冲数量，M为距离向输出点数（补零所需要）
__global__ void convertComplex(const float* datain_real,const float* datain_img,cuFloatComplex* complexdata,size_t nrn,size_t nan,size_t M)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    cuFloatComplex zero = make_cuComplex(0.0f,0.0f);
    if(i >= M*nan) return;
    if( i%M < nrn)
    {
        complexdata[i] = make_cuFloatComplex(datain_real[(i/M)*nrn + i%M],datain_img[(i/M)*nrn + i%M]);  
    }
    else
    {
        complexdata[i] = zero;
    }
}
//拆分复数同时截去补零
__global__ void splitComplex(const cuFloatComplex* complexdata, float* dataout_real, float* dataout_img,size_t M,size_t nrn,size_t nan)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if( i < M*nan)
    {
        if(i%M < nrn){
             dataout_real[(i/M)*nrn + i%M ] = complexdata[i].x;
             dataout_img[(i/M)*nrn + i%M] = complexdata[i].y;  
        }
    }
}


unsigned int nextPowerOfTwo(unsigned int n)
{
    --n;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    return n + 1;
}

void paluseCompression(float* datain_real,float* datain_img,float* dataout_real,float* dataout_img ,ProcessPara_t Para)
{
    cudaError_t ErroState;
    cufftResult cufftResult;
    float* dev_dataReal = nullptr;
    float* dev_dataImg = nullptr;
    cuFloatComplex* dev_datain = nullptr;
    cuFloatComplex* dev_dataout = nullptr;
    cuFloatComplex* dev_filter = nullptr;
    cuFloatComplex* dev_filterF = nullptr;
    cudaStream_t stream_1;
    cudaEvent_t start, stop;
    cufftHandle filterFftPlan,echoFftPlan;
    float timeRecord;
    int block =  512;
    int grid  = (Para.nan * Para.M + block - 1) / block;

    
    if (Para.nan <= 0 || Para.nrn <= 0) {
        fprintf(stderr, "nan or nrn is invalid\n");
        return;
    }
    //创建流和事件
    ErroState = cudaStreamCreate(&stream_1);
    if(ErroState != cudaSuccess) {
        fprintf(stderr, "CUDA stream creation failed: %s\n", cudaGetErrorString(ErroState));
        goto cleanup;
    }
    ErroState = cudaEventCreate(&start);
    if(ErroState != cudaSuccess) {
        fprintf(stderr, "CUDA event  start creation failed: %s\n", cudaGetErrorString(ErroState));
       goto cleanup;
    }
    ErroState = cudaEventCreate(&stop);
    if(ErroState != cudaSuccess) {
        fprintf(stderr, "CUDA event  stop creation failed: %s\n", cudaGetErrorString(ErroState));
        goto cleanup;
    }
    ErroState = cudaSetDevice(0);
    if (ErroState != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  ");
        return;
    }


    /*创建所有的FFT计划*/
    cufftPlan1d(&filterFftPlan,Para.M,CUFFT_C2C,1);
    if(ErroState != cudaSuccess) {
        fprintf(stderr, "CUDA create filter fft plan failed: %s\n", cudaGetErrorString(ErroState));
        goto cleanup;
    }
    cufftSetStream(filterFftPlan, stream_1);//将FFT计划与流关联
    //fft
    cufftPlan1d(&echoFftPlan,Para.M,CUFFT_C2C,Para.nan);
    if(ErroState != cudaSuccess) {
        fprintf(stderr, "CUDA create echo fft plan failed: %s\n", cudaGetErrorString(ErroState));
        goto cleanup;
    }
    cufftSetStream(echoFftPlan, stream_1);//将FFT计划与流关联

    /*分配GPU内存 */
    ErroState = cudaMalloc((void**)&dev_dataReal, Para.nan * Para.nrn * sizeof(float));
    if(ErroState != cudaSuccess) {
        fprintf(stderr, "CUDA memory allocation or copy failed: %s\n", cudaGetErrorString(ErroState));
        goto cleanup;   
    }
    ErroState = cudaMalloc((void**)&dev_dataImg, Para.nan * Para.nrn * sizeof(float));
    if(ErroState != cudaSuccess) {
        fprintf(stderr, "CUDA memory allocation or copy failed: %s\n", cudaGetErrorString(ErroState));
        goto cleanup;   
    }      
    ErroState = cudaMalloc((void**)&dev_datain, Para.nan * Para.M * sizeof(cuFloatComplex));
    if(ErroState != cudaSuccess) {
        fprintf(stderr, "CUDA memory allocation or copy failed: %s\n", cudaGetErrorString(ErroState));
        goto cleanup;   
    }
    ErroState = cudaMalloc((void**)&dev_dataout, Para.nan * Para.M * sizeof(cuFloatComplex));
    if(ErroState != cudaSuccess) {
        fprintf(stderr, "CUDA memory allocation or copy failed: %s\n", cudaGetErrorString(ErroState));
        goto cleanup;   
    }
    ErroState = cudaMalloc((void**)&dev_filter, Para.M * sizeof(cuFloatComplex));
    if(ErroState != cudaSuccess) {
        fprintf(stderr, "CUDA memory allocation or copy failed: %s\n", cudaGetErrorString(ErroState));
        goto cleanup;   
    }
    ErroState = cudaMalloc((void**)&dev_filterF, Para.M * sizeof(cuFloatComplex));
    if(ErroState != cudaSuccess) {
        fprintf(stderr, "CUDA memory allocation or copy failed: %s\n", cudaGetErrorString(ErroState));
        goto cleanup;   
    }
 /*将数据从CPU传输到GPU */
    ErroState = cudaMemcpy(dev_dataReal, datain_real, Para.nan * Para.nrn * sizeof(float), cudaMemcpyHostToDevice);       
    if(ErroState != cudaSuccess) {
        fprintf(stderr, "CUDA memory allocation or copy failed: %s\n", cudaGetErrorString(ErroState));
        goto cleanup;   
    }
    ErroState = cudaMemcpy(dev_dataImg, datain_img, Para.nan * Para.nrn * sizeof(float), cudaMemcpyHostToDevice);       
    if(ErroState != cudaSuccess) {
        fprintf(stderr, "CUDA memory allocation or copy failed: %s\n", cudaGetErrorString(ErroState));
        goto cleanup;   
    }
    /*GPU中执行核函数 */
    cudaEventRecord(start, stream_1);
    convertComplex<<<grid,block,0,stream_1>>>(dev_dataReal,dev_dataImg,dev_datain,Para.nrn,Para.nan,Para.M);
    ErroState = cudaGetLastError();
    if(ErroState != cudaSuccess) {
        fprintf(stderr, "Start CUDA kernel convertComplex failed: %s\n", cudaGetErrorString(ErroState));
        goto cleanup;   
    }   
    matchFilterInint<<<grid,block,0,stream_1>>>(Para,dev_filter,Para.M,Para.L);
    ErroState = cudaGetLastError();
    if(ErroState != cudaSuccess) {
        fprintf(stderr, "Start CUDA kernel “matchFilterInint” failed: %s\n", cudaGetErrorString(ErroState));
        goto cleanup;   
    }
    /*在GPU中执行核函数*/
    cufftResult = cufftExecC2C(filterFftPlan,dev_filter,dev_filterF,CUFFT_FORWARD);
    if(cufftResult != CUFFT_SUCCESS) {
        fprintf(stderr, "Start CUDA FFT failed: %s\n", cudaGetErrorString(ErroState));
        goto cleanup;   
    }
    cufftResult = cufftExecC2C(echoFftPlan,dev_datain,dev_dataout,CUFFT_FORWARD);
    if(cufftResult != CUFFT_SUCCESS) {
        fprintf(stderr, "Start CUDA ECHO FFT failed: %s\n", cudaGetErrorString(ErroState));
        goto cleanup;   
    }
    multiplyFilter<<<grid,block,0,stream_1>>>(dev_dataout,dev_datain, dev_filterF,Para.M,Para.nan);
    ErroState = cudaGetLastError();
    if(ErroState != cudaSuccess) {
        fprintf(stderr, "Start CUDA kernel “multiplyFilter” failed: %s\n", cudaGetErrorString(ErroState));
        goto cleanup;   
    }
    cufftResult = cufftExecC2C(echoFftPlan,dev_datain,dev_dataout,CUFFT_INVERSE);
    if(cufftResult != CUFFT_SUCCESS) {
        fprintf(stderr, "Start CUDA kernel “matchFilterInint” failed: %s\n", cudaGetErrorString(ErroState));
        goto cleanup;   
    }
    splitComplex<<<grid,block,0,stream_1>>>(dev_dataout, dev_dataReal, dev_dataImg, Para.M, Para.nrn, Para.nan);
    ErroState = cudaGetLastError();
    if(ErroState != cudaSuccess) {
        fprintf(stderr, "Start CUDA kernel splitComplex failed: %s\n", cudaGetErrorString(ErroState));
        goto cleanup;   
    }
    cudaEventRecord(stop, stream_1);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&timeRecord, start, stop);
    printf("Cuda设备完成该计算耗时：%f\n", timeRecord);

     /*将数据从GPU传输到CPU */
    ErroState = cudaMemcpy(dataout_real, dev_dataReal, Para.nan * Para.nrn * sizeof(float), cudaMemcpyDeviceToHost);       
    if(ErroState != cudaSuccess) {
        fprintf(stderr, "CUDA memory allocation or copy failed: %s\n", cudaGetErrorString(ErroState));
        goto cleanup;   
    }
    ErroState = cudaMemcpy(dataout_img, dev_dataImg, Para.nan * Para.nrn * sizeof(float), cudaMemcpyDeviceToHost);       
    if(ErroState != cudaSuccess) {
        fprintf(stderr, "CUDA memory allocation or copy failed: %s\n", cudaGetErrorString(ErroState));
        goto cleanup;   
    }
    
    cleanup:
    // 保留最先发生的错误，同时检查释放操作
    if (dev_filter) {
        cudaFree(dev_filter);
    }
    if (dev_dataout) {
        cudaFree(dev_dataout);
    }
    if (dev_datain) {
        cudaFree(dev_datain);
    }
    if (dev_filterF) {
        cudaFree(dev_filterF);
    }
    if (dev_dataReal) {
        cudaFree(dev_dataReal);
    }
    if (dev_dataImg) {
        cudaFree(dev_dataImg);
    }
    cudaStreamDestroy(stream_1);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cufftDestroy(filterFftPlan);
    cufftDestroy(echoFftPlan);
}