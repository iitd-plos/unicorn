
#ifdef BUILD_CUDA

#include "pmPublicDefinitions.h"
#include "prefixSum.h"

#include <iostream>

namespace prefixSum
{
    
#define NUM_BANKS 16  
#define LOG_NUM_BANKS 4  
#define CONFLICT_FREE_OFFSET(n) ((n) >> NUM_BANKS + (n) >> (2 * LOG_NUM_BANKS))  

__global__ void prefixSum_cuda(prefixSumTaskConf pTaskConf, pmSubtaskInfo pSubtaskInfo, void* pOutputBlock)
{
    PREFIX_SUM_DATA_TYPE* lInput = (PREFIX_SUM_DATA_TYPE*)(pSubtaskInfo.inputMem);
    PREFIX_SUM_DATA_TYPE* lOutput = (PREFIX_SUM_DATA_TYPE*)(pSubtaskInfo.outputMem);
    PREFIX_SUM_DATA_TYPE lCount = (unsigned int)((pSubtaskInfo.inputMemLength)/sizeof(PREFIX_SUM_DATA_TYPE));

    extern __shared__ float temp[];
    int thid =  blockIdx.x * blockDim.x + threadIdx.x;
    int offset = 1;

    int ai = thid;
    int bi = thid + (lCount/2);
    int bankOffsetA = CONFLICT_FREE_OFFSET(ai);
    int bankOffsetB = CONFLICT_FREE_OFFSET(bi);

    temp[ai + bankOffsetA] = lInput[ai];
    temp[bi + bankOffsetB] = lInput[bi];

    for(int d = lCount>>1; d > 0; d >>= 1)
    {   
        __syncthreads();

        if(thid < d)
        {  
            int ai = offset * (2 * thid + 1) - 1;
            int bi = offset * (2 * thid + 2) - 1;
            
            ai += CONFLICT_FREE_OFFSET(ai);
            bi += CONFLICT_FREE_OFFSET(bi);
        
           temp[bi] += temp[ai];
        }
        
        offset *= 2;
    }
    
    if(thid == 0)
        temp[lCount - 1 + CONFLICT_FREE_OFFSET(lCount - 1)] = 0;
        
    for(int d = 1; d < lCount; d *= 2)
    {  
         offset >>= 1;
         __syncthreads();
        
         if(thid < d)
         {  
            int ai = offset * (2 * thid + 1) - 1;
            int bi = offset * (2 * thid + 2) - 1;
            ai += CONFLICT_FREE_OFFSET(ai);
            bi += CONFLICT_FREE_OFFSET(bi);
             
            float t = temp[ai];
            temp[ai] = temp[bi];  
            temp[bi] += t;   
        }
    }

    __syncthreads();

    lOutput[ai] = temp[ai + bankOffsetA];
    lOutput[bi] = temp[bi + bankOffsetB];
}

__global__ void elemAdd_cuda(prefixSumTaskConf pTaskConf, pmSubtaskInfo pSubtaskInfo, void* pOutputBlock)
{
    PREFIX_SUM_DATA_TYPE lElem = ((PREFIX_SUM_DATA_TYPE*)(pSubtaskInfo.inputMem))[0];
    unsigned int lLength = (unsigned int)((pSubtaskInfo.outputMemLength)/sizeof(PREFIX_SUM_DATA_TYPE));
    
    unsigned int lThreadIndex = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int lStartIndex = lThreadIndex * ELEMS_ADD_PER_CUDA_THREAD;
    unsigned int lEndIndex = lStartIndex + ELEMS_ADD_PER_CUDA_THREAD;
    if(lEndIndex > lLength)
        lEndIndex = lLength;

    for(unsigned int i = lStartIndex; i < lEndIndex; ++i)
        ((PREFIX_SUM_DATA_TYPE*)(pSubtaskInfo.outputMem))[i] += lElem;

	return pmSuccess;
}
    
#endif


