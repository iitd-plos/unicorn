
#include "pmPublicDefinitions.h"
#include "matrixMultiply.h"

__global__ void matrixMultiply_cuda(pmTaskInfo pTaskInfo, pmSubtaskInfo pSubtaskInfo, pmStatus* pStatus)
{
    // Each thread computes one element of resulting matrix by accumulating results into value

    matrixMultiplyTaskConf* lTaskConf = (matrixMultiplyTaskConf*)(pTaskInfo.taskConf);
    unsigned int lDimension = lTaskConf->matrixDim;
    unsigned int lMatrixSize = lDimension * lDimension;


    MATRIX_DATA_TYPE value = (MATRIX_DATA_TYPE)0;
    int rowId = pSubtaskInfo.subtaskId;
    int colId = ((blockIdx.x * gridDim.y + blockIdx.y)*(blockDim.x * blockDim.y)) + (threadIdx.x * blockDim.y) + threadIdx.y;

    if(colId >= lDimension)
        return;

    MATRIX_DATA_TYPE* inputMem1 = (MATRIX_DATA_TYPE*)(pSubtaskInfo.inputMem);
    MATRIX_DATA_TYPE* inputMem2 = inputMem1 + lMatrixSize;
    MATRIX_DATA_TYPE* outputMem = (MATRIX_DATA_TYPE*)(pSubtaskInfo.outputMem);

    for(int e = 0; e < lDimension; ++e)
	value += inputMem1[rowId * lDimension + e] * inputMem2[e * lDimension + colId];

    outputMem[colId] = value;

    *pStatus = pmSuccess;
}
