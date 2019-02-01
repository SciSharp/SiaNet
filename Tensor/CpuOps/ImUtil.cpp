#include "ImUtil.h"
#include "TensorIter-inl.h"
#include "TensorApplyDim-inl.h"
#include "TensorApply-inl.h"
#include "math.h"
#include <thread>
#include <omp.h>
#include <iostream>

using namespace std;

int IM2COL_MAX_THREAD_NUMBER = (int)std::thread::hardware_concurrency() * .7;

typedef struct {
	float*  data_im;
	int     channels;
	int     height;
	int     width;
	int     kernel_h;
	int     kernel_w;
	int     pad_h;
	int     pad_w;
	int     stride_h;
	int     stride_w;
	int     dilation_h;
	int     dilation_w;
	int     output_h;
	int     output_w;
	int     channel_size;
	int     data_col_size;
	float*  data_col;
	int*    range_channel;
} im2col_arg_t;

inline bool is_a_ge_zero_and_a_lt_b(int a, int b) {
	return static_cast<unsigned>(a) < static_cast<unsigned>(b);
}

void im2col_inner_thread(void* set_args, int pos)
{
	im2col_arg_t* args = (im2col_arg_t*)set_args;
	float*  data_im = args->data_im;
	//int     channels      = args -> channels     ;
	int     height = args->height;
	int     width = args->width;
	int     kernel_h = args->kernel_h;
	int     kernel_w = args->kernel_w;
	int     pad_h = args->pad_h;
	int     pad_w = args->pad_w;
	int     stride_h = args->stride_h;
	int     stride_w = args->stride_w;
	int     dilation_h = args->dilation_h;
	int     dilation_w = args->dilation_w;
	int     output_h = args->output_h;
	int     output_w = args->output_w;
	int     channel_size = args->channel_size;
	int     data_col_size = args->data_col_size;
	float*  data_col = args->data_col;
	int*    range_channel = args->range_channel;

	data_im += range_channel[pos] * channel_size;
	data_col += range_channel[pos] * data_col_size;
	int channel, kernel_row, kernel_col, output_rows, output_cols;

	for (channel = range_channel[pos]; channel < range_channel[pos + 1]; channel++, data_im += channel_size)
	{
		for (kernel_row = 0; kernel_row < kernel_h; kernel_row++) {
			for (kernel_col = 0; kernel_col < kernel_w; kernel_col++) {
				int input_row = -pad_h + kernel_row * dilation_h;
				for (output_rows = output_h; output_rows; output_rows--) {
					if (!is_a_ge_zero_and_a_lt_b(input_row, height)) {
						for (output_cols = output_w; output_cols; output_cols--) {
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
	}
}

void col2im_inner_thread(void* set_args, int pos)
{
	im2col_arg_t* args = (im2col_arg_t*)set_args;

	float*  data_im = args->data_im;
	//int     channels      = args -> channels     ;
	int     height = args->height;
	int     width = args->width;
	int     kernel_h = args->kernel_h;
	int     kernel_w = args->kernel_w;
	int     pad_h = args->pad_h;
	int     pad_w = args->pad_w;
	int     stride_h = args->stride_h;
	int     stride_w = args->stride_w;
	int     dilation_h = args->dilation_h;
	int     dilation_w = args->dilation_w;
	int     output_h = args->output_h;
	int     output_w = args->output_w;
	int     channel_size = args->channel_size;
	int     data_col_size = args->data_col_size;
	float*  data_col = args->data_col;
	int*    range_channel = args->range_channel;

	data_im += range_channel[pos] * channel_size;
	data_col += range_channel[pos] * data_col_size;

	for (int channel = range_channel[pos]; channel < range_channel[pos + 1]; channel++, data_im += channel_size)
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
	}
}

static void divide(int M, int* range_M)
{
	int dx = M % IM2COL_MAX_THREAD_NUMBER;
	int dy = M / IM2COL_MAX_THREAD_NUMBER;
	int index = 0;
	int i;
	for (i = 0; i < IM2COL_MAX_THREAD_NUMBER + 1; i++)
	{
		range_M[i] = index;
		if (i < dx)
		{
			index = index + dy + 1;
		}
		else
		{
			index = index + dy;
		}
	}
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
	const int output_h = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
	const int output_w = (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
	const int channel_size = height * width;
	const int data_col_size = output_h * output_w * kernel_h * kernel_w;
	int channel;
	
	im2col_arg_t      ins_args;
	int* range_channel;
	
	ins_args.data_im = (float*)data_im;
	//ins_args.channels      = channels;
	ins_args.height = height;
	ins_args.width = width;
	ins_args.kernel_h = kernel_h;
	ins_args.kernel_w = kernel_w;
	ins_args.pad_h = pad_h;
	ins_args.pad_w = pad_w;
	ins_args.stride_h = stride_h;
	ins_args.stride_w = stride_w;
	ins_args.dilation_h = dilation_h;
	ins_args.dilation_w = dilation_w;
	ins_args.output_h = output_h;
	ins_args.output_w = output_w;
	ins_args.channel_size = channel_size;
	ins_args.data_col_size = data_col_size;
	ins_args.data_col = (float*)data_col;
	ins_args.range_channel = range_channel;
	
	divide(channels, range_channel);
	int i;
	omp_set_num_threads(IM2COL_MAX_THREAD_NUMBER);
#pragma omp parallel for
	for (i = 0; i < IM2COL_MAX_THREAD_NUMBER; i++)
	{
		im2col_inner_thread(&ins_args, i);
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
	const int data_col_size = output_h * output_w * kernel_h * kernel_w;
	int channel;

	im2col_arg_t      ins_args;
	int* range_channel;
	
	ins_args.data_im = (float*)data_im;
	//ins_args.channels      = channels;
	ins_args.height = height;
	ins_args.width = width;
	ins_args.kernel_h = kernel_h;
	ins_args.kernel_w = kernel_w;
	ins_args.pad_h = pad_h;
	ins_args.pad_w = pad_w;
	ins_args.stride_h = stride_h;
	ins_args.stride_w = stride_w;
	ins_args.dilation_h = dilation_h;
	ins_args.dilation_w = dilation_w;
	ins_args.output_h = output_h;
	ins_args.output_w = output_w;
	ins_args.channel_size = channel_size;
	ins_args.data_col_size = data_col_size;
	ins_args.data_col = (float*)data_col;
	ins_args.range_channel = range_channel;

	divide(channels, range_channel);
	int i;
	omp_set_num_threads(IM2COL_MAX_THREAD_NUMBER);
#pragma omp parallel for
	for (i = 0; i < IM2COL_MAX_THREAD_NUMBER; i++)
	{
		col2im_inner_thread(&ins_args, i);
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