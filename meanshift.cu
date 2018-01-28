/*Antoniadis Moschos, AEM = 8761, AUTH 2018 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include <sys/time.h>
#include <float.h>
#include <cuda_runtime.h>

#define EPSILON 0.0000001
#define MAX_ITERATIONS 30
#define N 2
#define SIGMA 1
#define BLOCK_SIZE 128
#define INPUT_FILE "x.bin"
#define RESULTS_FILE "results.txt"
#define gpuErrchk(ans)                        \
    {                                         \
        gpuAssert((ans), __FILE__, __LINE__); \
    }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort)
            exit(code);
    }
}

//__device__ const int BLOCK_SIZE = 640;

/********* Device Function - Gaussian ******/
__device__ double gaussian(double norm)
{

    return exp(-norm / (2 * SIGMA * SIGMA));
}

/******** Kernel Function *******/
__global__ void meanshiftKernel(double *devX, double *devY, int size)
{
    // __shared__ double devY[BLOCK_SIZE*N];
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < size) //if idx bigger than matrix height return
    {               //printf(" idx = %d devX = %f \n", idx,devX_[idx]);

        int j, k, iterations = 0;
        double sumNum[N] = {0}, sumDenum = 0, dist, meanshift = 0xFFFFFFFF, m_new[N], m[N];

        /* for (int currentBlockOfPoints = 0; currentBlockOfPoints < gridDim.x; currentBlockOfPoints++)
    {
        if (threadIdx.x + currentBlockOfPoints * BLOCK_SIZE  < size * N)
            for(k=0;k<N;k++)
            devY[threadIdx.x*N+k] = devX[threadIdx.x*N + k + (currentBlockOfPoints * BLOCK_SIZE)];
    }
    __syncthreads();
    */
        //if(idx == 599) printf(" idx = %d  devX= %.6f , %.6f\n", idx, devX[threadIdx.x*N], devX[threadIdx.x*N+1]);
        for (k = 0; k < N; k++)
            m[k] = devY[idx * N + k];
        //  __syncthreads();
        while (meanshift > EPSILON && iterations < MAX_ITERATIONS)
        {
            meanshift = 0;

            for (k = 0; k < N; k++)
                sumNum[k] = 0;
            sumDenum = 0;

            for (j = 0; j < size; j++)
            {
                dist = 0;
                for (k = 0; k < N; k++)
                    dist += pow(m[k] - devX[j * N + k], 2);
                if (dist < SIGMA) //dist is already squared
                {
                    for (k = 0; k < N; k++)
                        sumNum[k] += gaussian(dist) * devX[j * N + k];
                    sumDenum += gaussian(dist);
                }
            }
            for (k = 0; k < N; k++)
            {
                m_new[k] = sumNum[k] / sumDenum;

                meanshift += pow(m_new[k] - m[k], 2);

                m[k] = m_new[k];
            }
            meanshift = sqrt(meanshift);
            iterations++;

            if (idx == 0)
                printf("Iteration %d  error = %.9f \n", iterations, meanshift); //print iterations for first point
        }
        for (k = 0; k < N; k++)
        {
            devY[idx * N + k] = m[k];
        }

        /*for (int currentBlockOfPoints = 0; currentBlockOfPoints < gridDim.x; currentBlockOfPoints++)
    {
        if (threadIdx.x + currentBlockOfPoints * BLOCK_SIZE  < size * N)
            for(k=0;k<N;k++)
            devY_[threadIdx.x*N + k + (currentBlockOfPoints * BLOCK_SIZE)] = devY[threadIdx.x*N+k];
    }*/
        // __syncthreads();
        // printf(" idx = %d  devX= %.6f , %.6f\n", idx, devY[idx*N], devY[idx*N+1]);
    }
}

/******** Main ********/

int main(int argc, char **argv)
{
    FILE *inFile, *resultsFile;
    int filesize;
    size_t totalsize;
    double *x, *y;         // host matrices
    double *dev_x, *dev_y; // GPU matrices
    cudaEvent_t start, stop;
    float elapsedTime;
    /*Input file open */
    if ((inFile = fopen(INPUT_FILE, "rb+")) == NULL)
    {
        printf("\nFile not found\n");
        exit(1);
    }

    /* Get the size of the file */
    fseek(inFile, 0L, SEEK_END);
    filesize = ftell(inFile) / (sizeof(double) * N); //find the number of points
    rewind(inFile);
    totalsize = filesize * N * sizeof(double); //size in bytes
    printf("\nTotal points = %d  Dimensions = %d\n", filesize, N);

    /*Malloc for input buffer*/
    if ((x = (double *)malloc(totalsize)) == NULL)
        exit(1);

    /*Malloc for output buffer*/
    if ((y = (double *)malloc(totalsize)) == NULL)
        exit(1);

    /*Read the data to buffer*/
    if ((fread(x, sizeof(double), filesize * N, inFile)) != filesize * N)
    {
        fprintf(stderr, "Unable to read data\n");
        exit(1);
    }
    fclose(inFile);
    //memcpy(y, x,totalsize); //not needed

    /*Malloc for GPU input buffer*/
    gpuErrchk(cudaMalloc((void **)&dev_x, totalsize));

    /*Malloc for output buffer*/
    gpuErrchk(cudaMalloc((void **)&dev_y, totalsize));

    /*Copy data to GPU global memory*/
    gpuErrchk(cudaMemcpy(dev_x, x, totalsize, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(dev_y, x, totalsize, cudaMemcpyHostToDevice));

    //  dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    //  dim3 dimGrid(filesize / dimBlock.x, filesize / dimBlock.y);
    int threadsPerBlock = BLOCK_SIZE;
    int blocksPerGrid = (filesize + threadsPerBlock - 1) / threadsPerBlock;

    cudaEventCreate(&start);
    cudaEventRecord(start, 0);

    meanshiftKernel<<<blocksPerGrid, threadsPerBlock>>>(dev_x, dev_y, filesize);

    cudaEventCreate(&stop);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Elapsed time : %f ms\n", elapsedTime);
    cudaDeviceSynchronize();
    gpuErrchk(cudaMemcpy(y, dev_y, totalsize, cudaMemcpyDeviceToHost));
    cudaDeviceSynchronize();

    /*
    Print Data

    for(int i=0; i<filesize; i++)
    {
        printf("%.9f %.9f \n",y[i*N],y[i*N+1]);
    }*/
    cudaFree(dev_x);
    cudaFree(dev_y);

    resultsFile = fopen(RESULTS_FILE, "wb");
    for (int i = 0; i < filesize; i++)
    {
        for (int k = 0; k < N; k++)
            fprintf(resultsFile, "%.6f\t", y[i * N + k]);
        fprintf(resultsFile, "\n");
    }
    fclose(resultsFile);
    free(x);
    free(y);
}