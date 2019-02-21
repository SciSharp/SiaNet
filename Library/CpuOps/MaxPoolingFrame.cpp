#include "MaxPoolingFrame.h"


int TS_SpatialMaxPooling_updateOutput_frame(
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
	API_BEGIN()
	SWITCH_TENSOR_TYPE_ALL_CPU(input->elementType, SpatialMaxPooling_updateOutput_frame, input, output, ind, nslices, iwidth, iheight, owidth, oheight, kW, kH, dW, dH, padW, padH)
	API_END()
}

int TS_SpatialMaxPooling_updateGradInput_frame(
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
	API_BEGIN()
	SWITCH_TENSOR_TYPE_ALL_CPU(gradInput->elementType, SpatialMaxPooling_updateGradInput_frame, gradInput, gradOutput, ind, nslices, iwidth, iheight, owidth, oheight, dW, dH)
	API_END()
}
