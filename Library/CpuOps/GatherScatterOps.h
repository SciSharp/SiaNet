#pragma once

#include "General.h"
#include "TensorRef.h"


OPS_API int TS_Gather(TensorRef* result, TensorRef* src, int dim, TensorRef* indices);
OPS_API int TS_Scatter(TensorRef* result, TensorRef* src, int dim, TensorRef* indices);
OPS_API int TS_ScatterFill(TensorRef* result, float value, int dim, TensorRef* indices);
OPS_API int TS_Diag(TensorRef* result, TensorRef* src);