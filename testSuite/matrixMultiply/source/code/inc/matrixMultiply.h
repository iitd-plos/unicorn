
#define DEFAULT_MATRIX_DIM 1000
#define MATRIX_DATA_TYPE int

pmStatus matrixMultiply_cuda(pmTaskInfo pTaskInfo, pmSubtaskInfo pSubtaskInfo);

typedef struct matrixMultiplyTaskConf
{
	int matrixDim;
} matrixMultiplyTaskConf;