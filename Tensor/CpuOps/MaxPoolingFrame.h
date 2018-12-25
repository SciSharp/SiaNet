#pragma once

#include <algorithm>
#include <climits>


#include "General.h"
#include "TensorRef.h"


OPS_API int TS_SpatialMaxPooling_updateOutput_frame(
	TensorRef *input_p,
	TensorRef *output_p,
	TensorRef *ind_p,
	__int64 nslices,
	__int64 iwidth,
	__int64 iheight,
	__int64 owidth,
	__int64 oheight,
	int kW,
	int kH,
	int dW,
	int dH,
	int padW,
	int padH);

OPS_API int TS_SpatialMaxPooling_updateGradInput_frame(
	TensorRef *gradInput,
	TensorRef *gradOutput,
	TensorRef *ind,
	__int64 nslices,
	__int64 iwidth,
	__int64 iheight,
	__int64 owidth,
	__int64 oheight,
	int dW,
	int dH);


template<typename T>
void SpatialMaxPooling_updateOutput_frame(
	TensorRef *input,
	TensorRef *output,
	TensorRef *ind,
	__int64 nslices,
	__int64 iwidth,
	__int64 iheight,
	__int64 owidth,
	__int64 oheight,
	int kW,
	int kH,
	int dW,
	int dH,
	int padW,
	int padH)
{
	T* input_p = (T*)input->buffer;
	T* output_p = (T*)output->buffer;
	T* ind_p = (T*)ind->buffer;

	__int64 k;
#pragma omp parallel for private(k)
	for (k = 0; k < nslices; k++)
	{
		/* loop over output */
		__int64 i, j;
		T *ip = input_p + k*iwidth*iheight;
		for (i = 0; i < oheight; i++)
		{
			for (j = 0; j < owidth; j++)
			{
				__int64 hstart = i * dH - padH;
				__int64 wstart = j * dW - padW;
				__int64 hend = std::min(hstart + kH, iheight);
				__int64 wend = std::min(wstart + kW, iwidth);
				hstart = std::max(hstart, 0LL);
				wstart = std::max(wstart, 0LL);

				/* local pointers */
				T *op = output_p + k*owidth*oheight + i*owidth + j;
				T *indp = ind_p + k*owidth*oheight + i*owidth + j;

				/* compute local max: */
				__int64 maxindex = -1;
				T maxval = std::numeric_limits<T>::min();
				__int64 tcntr = 0;
				__int64 x, y;
				for (y = hstart; y < hend; y++)
				{
					for (x = wstart; x < wend; x++)
					{
						tcntr = y*iwidth + x;
						T val = *(ip + tcntr);
						if (val > maxval)
						{
							maxval = val;
							maxindex = tcntr;
						}
					}
				}

				/* set output to local max */
				*op = maxval;

				/* store location of max */
				*indp = T(maxindex + 1);
			}
		}
	}
}

template<typename T>
void SpatialMaxPooling_updateGradInput_frame(
	TensorRef *gradInput,
	TensorRef *gradOutput,
	TensorRef *ind,
	__int64 nslices,
	__int64 iwidth,
	__int64 iheight,
	__int64 owidth,
	__int64 oheight,
	int dW,
	int dH)
{
	T* gradInput_p = (T*)gradInput->buffer;
	T* gradOutput_p = (T*)gradOutput->buffer;
	T* ind_p = (T*)ind->buffer;

	__int64 k;
#pragma omp parallel for private(k)
	for (k = 0; k < nslices; k++)
	{
		T *gradInput_p_k = gradInput_p + k*iwidth*iheight;
		T *gradOutput_p_k = gradOutput_p + k*owidth*oheight;
		T *ind_p_k = ind_p + k*owidth*oheight;

		/* calculate max points */
		__int64 i, j;
		for (i = 0; i < oheight; i++)
		{
			for (j = 0; j < owidth; j++)
			{
				/* retrieve position of max */
				__int64 maxp = __int64(ind_p_k[i*owidth + j] - 1);
				/* update gradient */
				gradInput_p_k[maxp] += gradOutput_p_k[i*owidth + j];
			}
		}
	}
}