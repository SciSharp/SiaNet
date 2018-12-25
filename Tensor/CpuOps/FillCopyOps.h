#pragma once

#include "General.h"
#include "TensorRef.h"

OPS_API int TS_Fill(TensorRef* result, float value);
OPS_API int TS_Copy(TensorRef* result, TensorRef* src);
