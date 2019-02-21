using System;
using System.Collections.Generic;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Text;
using SiaNet.Backend.TensorSharp.Core;

namespace SiaNet.Backend.TensorSharp.Cpu
{
    public class Im2ColCpu
    {
        private MethodInfo im2cols_func = NativeWrapper.GetMethod("TS_Im2Cols");
     
        [RegisterOpStorageType("im2cols", typeof(CpuStorage))]
        public void Im2Cols(NDArray im, NDArray col, int channels,
                            int height, int width,
                            int ksize_h, int ksize_w, int pad_h,
                            int pad_w, int stride_h, int stride_w,
                            int dilation_h, int dilation_w)
        {
            int height_col = (height + 2 * pad_h - (dilation_h * (ksize_h - 1) + 1))
                             / stride_h + 1;
            int width_col = (width + 2 * pad_w - (dilation_w * (ksize_w - 1) + 1))
                            / stride_w + 1;

            NativeWrapper.InvokeTypeMatch(im2cols_func, im, height, width, channels, ksize_h, ksize_w, pad_h, pad_w, stride_h, stride_w,
                                            dilation_h, dilation_w, height_col, width_col, col);
        }

        private MethodInfo cols2im_func = NativeWrapper.GetMethod("TS_Cols2Im");

        [RegisterOpStorageType("im2cols", typeof(CpuStorage))]
        public void Cols2Im(NDArray col, NDArray im, int channels, int height, int width,
                                int patch_h, int patch_w, int pad_h,
                                int pad_w, int stride_h, int stride_w,
                                int dilation_h, int dilation_w)
        {
            int height_col = (height + 2 * pad_h - (dilation_h * (patch_h - 1) + 1))
                   / stride_h + 1;
            int width_col = (width + 2 * pad_w - (dilation_w * (patch_w - 1) + 1))
                             / stride_w + 1;

            NativeWrapper.InvokeTypeMatch(cols2im_func, col, height, width, channels, patch_h, patch_w, pad_h, pad_w, stride_h, stride_w,
                                            dilation_h, dilation_w, height_col, width_col, im);
        }
    }
}
