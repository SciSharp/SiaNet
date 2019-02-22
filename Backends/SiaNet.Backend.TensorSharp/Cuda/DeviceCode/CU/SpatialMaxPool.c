#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

// kernels borrowed from Caffe
template <typename Dtype>
__global__ void MaxPoolForward_t(const int nthreads, const Dtype* bottom_data,
	const int num, const int channels, const int height,
	const int width, const int pooled_height, const int pooled_width,
	const int kernel_h, const int kernel_w, const int stride_h,
	const int stride_w, const int pad_h, const int pad_w, Dtype* top_data,
	Dtype* top_mask, Dtype minVal) {
	CUDA_KERNEL_LOOP(index, nthreads) {
		int pw = index % pooled_width;
		int ph = (index / pooled_width) % pooled_height;
		int c = (index / pooled_width / pooled_height) % channels;
		int n = index / pooled_width / pooled_height / channels;
		int hstart = ph * stride_h - pad_h;
		int wstart = pw * stride_w - pad_w;
		int hend = min(hstart + kernel_h, height);
		int wend = min(wstart + kernel_w, width);
		hstart = max(hstart, 0);
		wstart = max(wstart, 0);
		Dtype maxval = minVal;
		int maxidx = -1;
		bottom_data += (n * channels + c) * height * width;
		for (int h = hstart; h < hend; ++h) {
			for (int w = wstart; w < wend; ++w) {
				if (bottom_data[h * width + w] > maxval) {
					maxidx = h * width + w;
					maxval = bottom_data[maxidx];
				}
			}
		}
		top_data[index] = maxval;
		top_mask[index] = maxidx + 1;
	}
}


template <typename Dtype>
__global__ void MaxPoolBackward_t(const int nthreads, const Dtype* top_diff,
	const Dtype* top_mask, const int num, const int channels,
	const int height, const int width, const int pooled_height,
	const int pooled_width, const int kernel_h, const int kernel_w,
	const int stride_h, const int stride_w, const int pad_h, const int pad_w,
	Dtype* bottom_diff) {
	CUDA_KERNEL_LOOP(index, nthreads) {
		// find out the local index
		// find out the local offset
		int w = index % width;
		int h = (index / width) % height;
		int c = (index / width / height) % channels;
		int n = index / width / height / channels;
		int phstart =
			(h + pad_h < kernel_h) ? 0 : (h + pad_h - kernel_h) / stride_h + 1;
		int phend = min((h + pad_h) / stride_h + 1, pooled_height);
		int pwstart =
			(w + pad_w < kernel_w) ? 0 : (w + pad_w - kernel_w) / stride_w + 1;
		int pwend = min((w + pad_w) / stride_w + 1, pooled_width);
		Dtype gradient = 0;
		int offset = (n * channels + c) * pooled_height * pooled_width;
		top_diff += offset;
		top_mask += offset;
		for (int ph = phstart; ph < phend; ++ph) {
			for (int pw = pwstart; pw < pwend; ++pw) {
				if (top_mask[ph * pooled_width + pw] - 1 == h * width + w) {
					gradient += top_diff[ph * pooled_width + pw];
				}
			}
		}
		bottom_diff[index] = gradient;
	}
}


extern "C" {

#define FLT_MIN 1.17549435e-38F

	__global__ void MaxPoolForward(const int nthreads, const float* bottom_data,
		const int num, const int channels, const int height,
		const int width, const int pooled_height, const int pooled_width,
		const int kernel_h, const int kernel_w, const int stride_h,
		const int stride_w, const int pad_h, const int pad_w, float* top_data,
		float* top_mask)
	{
		MaxPoolForward_t(nthreads, bottom_data, num, channels, height, width, pooled_height, pooled_width, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, top_data, top_mask, FLT_MIN);
	}

	__global__ void MaxPoolBackward(const int nthreads, const float* top_diff,
		const float* top_mask, const int num, const int channels,
		const int height, const int width, const int pooled_height,
		const int pooled_width, const int kernel_h, const int kernel_w,
		const int stride_h, const int stride_w, const int pad_h, const int pad_w,
		float* bottom_diff)
	{
		MaxPoolBackward_t(nthreads, top_diff, top_mask, num, channels, height, width, pooled_height, pooled_width, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, bottom_diff);
	}

}