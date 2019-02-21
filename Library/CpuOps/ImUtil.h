#pragma once

#include "General.h"
#include "TensorRef.h"
#include "TensorApply-inl.h"

OPS_API int TS_Im2Cols(TensorRef* data_im,
	int height, int width, int channels,
	int ksize_h, int ksize_w,
	int pad_h, int pad_w,
	int stride_h, int stride_w,
	int dilation_h, int dilation_w,
	int height_col, int width_col,
	TensorRef* data_col);

OPS_API int TS_Cols2Im(TensorRef* data_col,
	int height, int width, int channels,
	int kernel_h, int kernel_w,
	int pad_h, int pad_w,
	int stride_h, int stride_w,
	int dilation_h, int dilation_w,
	int height_col, int width_col,
	TensorRef* data_im);

