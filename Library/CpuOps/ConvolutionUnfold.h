#pragma once

#include <string.h>
#include <math.h>
#include <algorithm>

#include "General.h"
#include "TensorRef.h"
#include "Vector-inl.h"


OPS_API int TS_Unfolded_Copy(
	TensorRef* finput,
	TensorRef* input,
	int kW,
	int kH,
	int dW,
	int dH,
	int padW,
	int padH,
	int nInputPlane,
	int inputWidth,
	int inputHeight,
	int outputWidth,
	int outputHeight);

OPS_API int TS_Unfolded_Acc(
	TensorRef *finput,
	TensorRef *input,
	int kW,
	int kH,
	int dW,
	int dH,
	int padW,
	int padH,
	int nInputPlane,
	int inputWidth,
	int inputHeight,
	int outputWidth,
	int outputHeight);


// note: due to write issues, this one cannot be parallelized as well as unfolded_copy
template<typename T>
void unfolded_acc(
	TensorRef *finput,
	TensorRef *input,
	int kW,
	int kH,
	int dW,
	int dH,
	int padW,
	int padH,
	int nInputPlane,
	int inputWidth,
	int inputHeight,
	int outputWidth,
	int outputHeight)
{
	size_t nip;

	T *input_data = (T*)input->buffer;
	T *finput_data = (T*)finput->buffer;

#pragma omp parallel
	for (nip = 0; nip < nInputPlane; nip++)
	{
		size_t kw, kh, y, x;
		__int64 ix = 0, iy = 0;
		for (kh = 0; kh < kH; kh++)
		{
			for (kw = 0; kw < kW; kw++)
			{
				T *src = finput_data + nip*(kH*kW*outputHeight*outputWidth) + kh*(kW*outputHeight*outputWidth) + kw*(outputHeight*outputWidth);
				T *dst = input_data + nip*(inputHeight*inputWidth);
				if (padW > 0 || padH > 0) {
					size_t lpad, rpad;
					for (y = 0; y < outputHeight; y++) {
						iy = (__int64)(y*dH - padH + kh);
						if (iy < 0 || iy >= inputHeight) {
						}
						else {
							if (dW == 1) {
								ix = (__int64)(0 - padW + kw);
								lpad = std::max(size_t(0), (padW - kw));
								rpad = std::max(size_t(0), (padW - (kW - kw - 1)));
								Vector_add<T>(dst + (size_t)(iy*inputWidth + ix + lpad), src + (size_t)(y*outputWidth + lpad), 1, outputWidth - lpad - rpad);
							}
							else {
								for (x = 0; x<outputWidth; x++) {
									ix = (__int64)(x*dW - padW + kw);
									if (ix < 0 || ix >= inputWidth) {
									}
									else
										Vector_add<T>(dst + (size_t)(iy*inputWidth + ix), src + (size_t)(y*outputWidth + x), 1, 1);
								}
							}
						}
					}
				}
				else {
					for (y = 0; y < outputHeight; y++) {
						iy = (__int64)(y*dH + kh);
						ix = (__int64)(0 + kw);
						if (dW == 1)
							Vector_add<T>(dst + (size_t)(iy*inputWidth + ix), src + (size_t)(y*outputWidth), 1, outputWidth);
						else {
							for (x = 0; x < outputWidth; x++)
								Vector_add<T>(dst + (size_t)(iy*inputWidth + ix + x*dW), src + (size_t)(y*outputWidth + x), 1, 1);
						}
					}
				}
			}
		}
	}
}



template<typename T>
void unfolded_copy(TensorRef *finput, TensorRef *input,
	int kW,
	int kH,
	int dW,
	int dH,
	int padW,
	int padH,
	int nInputPlane,
	int inputWidth,
	int inputHeight,
	int outputWidth,
	int outputHeight)
{
	long k;
	T *input_data = (T*)input->buffer;
	T *finput_data = (T*)finput->buffer;

#pragma omp parallel for private(k)
	for (k = 0; k < nInputPlane*kH*kW; k++) {
		size_t nip = k / (kH*kW);
		size_t rest = k % (kH*kW);
		size_t kh = rest / kW;
		size_t kw = rest % kW;
		size_t x, y;
		__int64 ix, iy;
		T *dst = finput_data + nip*(kH*kW*outputHeight*outputWidth) + kh*(kW*outputHeight*outputWidth) + kw*(outputHeight*outputWidth);
		T *src = input_data + nip*(inputHeight*inputWidth);
		if (padW > 0 || padH > 0) {
			size_t lpad, rpad;
			for (y = 0; y < outputHeight; y++) {
				iy = (__int64)(y*dH - padH + kh);
				if (iy < 0 || iy >= inputHeight) {
					memset(dst + y*outputWidth, 0, sizeof(T)*outputWidth);
				}
				else {
					if (dW == 1) {
						ix = (__int64)(0 - padW + kw);
						lpad = std::max(size_t(0), (padW - kw));
						rpad = std::max(size_t(0), (padW - (kW - kw - 1)));
						if (outputWidth - rpad - lpad <= 0) {
							memset(dst + (size_t)(y*outputWidth), 0, sizeof(T)*outputWidth);
						}
						else {
							if (lpad > 0) memset(dst + y*outputWidth, 0, sizeof(T)*lpad);
							memcpy(dst + (size_t)(y*outputWidth + lpad), src + (size_t)(iy*inputWidth + ix + lpad), sizeof(T)*(outputWidth - rpad - lpad));
							if (rpad > 0) memset(dst + y*outputWidth + outputWidth - rpad, 0, sizeof(T)*rpad);
						}
					}
					else {
						for (x = 0; x<outputWidth; x++) {
							ix = (__int64)(x*dW - padW + kw);
							if (ix < 0 || ix >= inputWidth)
								memset(dst + (size_t)(y*outputWidth + x), 0, sizeof(T) * 1);
							else
								memcpy(dst + (size_t)(y*outputWidth + x), src + (size_t)(iy*inputWidth + ix), sizeof(T)*(1));
						}
					}
				}
			}
		}
		else {
			for (y = 0; y < outputHeight; y++) {
				iy = (__int64)(y*dH + kh);
				ix = (__int64)(0 + kw);
				if (dW == 1)
					memcpy(dst + (size_t)(y*outputWidth), src + (size_t)(iy*inputWidth + ix), sizeof(T)*outputWidth);
				else {
					for (x = 0; x<outputWidth; x++)
						memcpy(dst + (size_t)(y*outputWidth + x), src + (size_t)(iy*inputWidth + ix + x*dW), sizeof(T)*(1));
				}
			}
		}
	}
}