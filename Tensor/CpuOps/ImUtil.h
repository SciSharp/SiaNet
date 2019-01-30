#pragma once

#include "General.h"
#include "TensorRef.h"

OPS_API int TS_Im2Cols(const int n, const TensorRef* data_im,
	const int height, const int width,
	const int ksize_h, const int ksize_w,
	const int pad_h, const int pad_w,
	const int stride_h, const int stride_w,
	const int dilation_h, const int dilation_w,
	const int height_col, const int width_col,
	TensorRef* data_col);

OPS_API int TS_Cols2Im(const int n, float* data_col,
	const int height, const int width, const int channels,
	const int kernel_h, const int kernel_w,
	const int pad_h, const int pad_w,
	const int stride_h, const int stride_w,
	const int dilation_h, const int dilation_w,
	const int height_col, const int width_col,
	float* data_im);

