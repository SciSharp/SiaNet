#include "ImUtil.h"
#include "TensorIter-inl.h"
#include "TensorApplyDim-inl.h"
#include "TensorApply-inl.h"
#include "math.h"
#include <thread>
#include <omp.h>
#include <iostream>

using namespace std;

inline bool is_a_ge_zero_and_a_lt_b(int a, int b) {
	return static_cast<unsigned>(a) < static_cast<unsigned>(b);
}

template<typename T>
INLINE_FUNC void im2cols(const TensorRef* data_im_t,
	const int height, const int width, const int channels,
	const int kernel_h, const int kernel_w,
	const int pad_h, const int pad_w,
	const int stride_h, const int stride_w,
	const int dilation_h, const int dilation_w,
	int height_col, int width_col,
	TensorRef* data_col_t) {
	T* data_im = (T*)data_im_t->buffer;
	T* data_col = (T*)data_col_t->buffer;
	int dil_kernel_h = (kernel_h - 1) * dilation_h + 1;
	int dil_kernel_w = (kernel_w - 1) * dilation_w + 1;
	height_col = (height + 2 * pad_h - dil_kernel_h) / stride_h + 1;
	width_col = (width + 2 * pad_w - dil_kernel_w) / stride_w + 1;
	int channels_col = channels * kernel_h * kernel_w;
#pragma omp parallel for
	for (int c = 0; c < channels_col; ++c) {
		int w_offset = c % kernel_w;
		int h_offset = (c / kernel_w) % kernel_h;
		int c_im = c / (kernel_h * kernel_w);

		const int hc0 = h_offset * dilation_h - pad_h;
		const int wc0 = w_offset * dilation_w - pad_w;
		for (int h = 0; h < height_col; ++h) {
			int h_pad = h * stride_h + hc0;

			const int row_offset = (c * height_col + h) * width_col;
			const int srow_offset = (c_im * height + h_pad) * width;
			for (int w = 0; w < width_col; ++w) {
				int w_pad = w * stride_w + wc0;
				if ((((unsigned)h_pad) < ((unsigned)height)) && (((unsigned)w_pad) < ((unsigned)width)))
					data_col[row_offset + w] = data_im[srow_offset + w_pad];
				else {
					data_col[row_offset + w] = 0.;
				}
			}
		}
	}
}



template<typename T>
INLINE_FUNC void cols2im(const TensorRef* data_col_t,
	const int height, const int width, const int channels,
	const int kernel_h, const int kernel_w,
	const int pad_h, const int pad_w,
	const int stride_h, const int stride_w,
	const int dilation_h, const int dilation_w,
	int height_col, int width_col,
	TensorRef* data_im_t) {
	T *data_im = (T*)data_im_t->buffer;
	T *data_col = (T*)data_col_t->buffer;

	int dil_patch_h = (kernel_h - 1) * dilation_h + 1;
	int dil_patch_w = (kernel_w - 1) * dilation_w + 1;
	height_col = (height + 2 * pad_h - dil_patch_h) / stride_h + 1;
	width_col = (width + 2 * pad_w - dil_patch_w) / stride_w + 1;
	long chunk_len = kernel_h * kernel_w;

	//caffe_set((size_t)height * (size_t)width * (size_t)channels, Dtype(0), data_im);

#pragma omp parallel for if (channels > 1)
	for (int idx = 0; idx < channels; ++idx) {
		for (int inner_idx = 0; inner_idx < chunk_len; ++inner_idx) {
			int c = idx * chunk_len + inner_idx;
			int w_offset = c % kernel_w;
			int h_offset = (c / kernel_w) % kernel_h;
			int c_im = c / (kernel_h * kernel_w);

			const int hc0 = h_offset * dilation_h - pad_h;
			const int wc0 = w_offset * dilation_w - pad_w;
			for (int h = 0; h < height_col; ++h) {
				for (int w = 0; w < width_col; ++w) {
					int h_pad = h * stride_h + hc0;
					const int srow_offset = (c_im * height + h_pad) * width;
					const int row_offset = (c * height_col + h) * width_col;
					int w_pad = w * stride_w + wc0;
					if ((((unsigned)h_pad) < ((unsigned)height)) && (((unsigned)w_pad) < ((unsigned)width))) {
						data_im[srow_offset + w_pad] += data_col[row_offset + w];
					}
				}
			}
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