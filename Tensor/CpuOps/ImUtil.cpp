#include "ImUtil.h"
#include "TensorIter-inl.h"
#include "TensorApplyDim-inl.h"
#include "TensorApply-inl.h"
#include "math.h"
#include <ppl.h>

using namespace concurrency;
using namespace std;

inline bool is_a_ge_zero_and_a_lt_b(int a, int b) {
	return static_cast<unsigned>(a) < static_cast<unsigned>(b);
}

template<typename T>
INLINE_FUNC void im2cols(TensorRef* data_im_t,
	int height, int width, int channels,
	int kernel_h, int kernel_w,
	int pad_h, int pad_w,
	int stride_h, int stride_w,
	int dilation_h, int dilation_w,
	int height_col, int width_col,
	TensorRef* data_col_t) {
	T* data_im = (T*)data_im_t->buffer;
	T* data_col = (T*)data_col_t->buffer;
	const int output_h = (height + 2 * pad_h -
		(dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
	const int output_w = (width + 2 * pad_w -
		(dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
	const int channel_size = height * width;
	int channel;
#pragma omp parallel
	{
#pragma omp for
		for (channel = 0; channel < channels; channel++)
		{
			for (int kernel_row = 0; kernel_row < kernel_h; kernel_row++) {
				for (int kernel_col = 0; kernel_col < kernel_w; kernel_col++) {
					int input_row = -pad_h + kernel_row * dilation_h;
					for (int output_rows = output_h; output_rows; output_rows--) {
						if (!is_a_ge_zero_and_a_lt_b(input_row, height)) {
							for (int output_cols = output_w; output_cols; output_cols--) {
								*(data_col++) = 0;
							}
						}
						else {
							int input_col = -pad_w + kernel_col * dilation_w;

							for (int output_col = output_w; output_col; output_col--) {
								if (is_a_ge_zero_and_a_lt_b(input_col, width)) {
									*(data_col++) = data_im[input_row * width + input_col];
								}
								else {
									*(data_col++) = 0;
								}
								input_col += stride_w;
							}
						}
						input_row += stride_h;
					}
				}
			}

			data_im += channel_size;
		}

	}
}

template<typename T>
INLINE_FUNC void cols2im(TensorRef* data_col_t,
	int height, int width, int channels,
	int kernel_h, int kernel_w,
	int pad_h, int pad_w,
	int stride_h, int stride_w,
	int dilation_h, int dilation_w,
	int height_col, int width_col,
	TensorRef* data_im_t) {
	T* data_im = (T*)data_im_t->buffer;
	T* data_col = (T*)data_col_t->buffer;

	const int output_h = (height + 2 * pad_h -
		(dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
	const int output_w = (width + 2 * pad_w -
		(dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
	const int channel_size = height * width;
	int channel;
#pragma omp parallel
	{
#pragma omp for
		for (channel = 0; channel < channels; channel++){
			for (int kernel_row = 0; kernel_row < kernel_h; kernel_row++) {
				for (int kernel_col = 0; kernel_col < kernel_w; kernel_col++) {
					int input_row = -pad_h + kernel_row * dilation_h;
					for (int output_rows = output_h; output_rows; output_rows--) {
						if (!is_a_ge_zero_and_a_lt_b(input_row, height)) {
							for (int output_cols = output_w; output_cols; output_cols--) {
								*(data_col++) = 0;
							}
						}
						else {
							int input_col = -pad_w + kernel_col * dilation_w;
							for (int output_col = output_w; output_col; output_col--) {
								if (is_a_ge_zero_and_a_lt_b(input_col, width)) {
									*(data_col++) = data_im[input_row * width + input_col];
								}
								else {
									*(data_col++) = 0;
								}
								input_col += stride_w;
							}
						}
						input_row += stride_h;
					}
				}
			}
			data_im += channel_size;
		}
	}
}

int TS_Im2Cols(TensorRef* data_im,
	int height, int width, int channels,
	int ksize_h, int ksize_w,
	int pad_h, int pad_w,
	int stride_h, int stride_w,
	int dilation_h, int dilation_w,
	int height_col, int width_col,
	TensorRef* data_col)
{
	API_BEGIN()
	SWITCH_TENSOR_TYPE_ALL_CPU(data_im->elementType, im2cols, data_im, height, width, channels, ksize_h, ksize_w, pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w, height_col, width_col, data_col)
	API_END()
}

int TS_Cols2Im(TensorRef* data_col,
	int height, int width, int channels,
	int kernel_h, int kernel_w,
	int pad_h, int pad_w,
	int stride_h, int stride_w,
	int dilation_h, int dilation_w,
	int height_col, int width_col,
	TensorRef* data_im)
{
	API_BEGIN()
	SWITCH_TENSOR_TYPE_ALL_CPU(data_col->elementType, cols2im, data_col, height, width, channels, kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w, height_col, width_col, data_im)
	API_END()
}