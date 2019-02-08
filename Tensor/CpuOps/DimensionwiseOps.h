#pragma once

#include "General.h"
#include "TensorRef.h"


OPS_API int TS_Sum(TensorRef* result, TensorRef* src, int dimension);
OPS_API int TS_Prod(TensorRef* result, TensorRef* src, int dimension);
OPS_API int TS_Min(TensorRef* result, TensorRef* src, int dimension);
OPS_API int TS_Max(TensorRef* result, TensorRef* src, int dimension);

OPS_API int TS_Argmin(TensorRef *resultIndices, TensorRef* src, int dimension);
OPS_API int TS_Argmax(TensorRef *resultIndices, TensorRef* src, int dimension);


OPS_API int TS_SumAll(TensorRef* result, TensorRef* src);
OPS_API int TS_ProdAll(TensorRef* result, TensorRef* src);
OPS_API int TS_MinAll(TensorRef* result, TensorRef* src);
OPS_API int TS_MaxAll(TensorRef* result, TensorRef* src);


//
// Floating point only ops
//
OPS_API int TS_Mean(TensorRef* result, TensorRef* src, int dimension);
OPS_API int TS_Norm(TensorRef* result, TensorRef* src, int dimension, float value);

// If normByN == true, result is normalized by n,
// otherwise, result is normalized by n-1
OPS_API int TS_Std(TensorRef* result, TensorRef* src, int dimension, bool normByN);
OPS_API int TS_Var(TensorRef* result, TensorRef* src, int dimension, bool normByN);



OPS_API int TS_MeanAll(TensorRef* result, TensorRef* src);
OPS_API int TS_VarAll(TensorRef* result, TensorRef* src);
OPS_API int TS_StdAll(TensorRef* result, TensorRef* src);
OPS_API int TS_NormAll(TensorRef* result, TensorRef* src, float value);
OPS_API int TS_Diag(TensorRef* result, TensorRef* src);


//
// End of floating-point only ops
//


