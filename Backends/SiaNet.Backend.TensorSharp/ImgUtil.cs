using System;
using System.Collections.Generic;
using System.Text;
using System.Linq;
using SiaNet.Backend.TensorSharp;
using SiaNet.Backend.TensorSharp.CUDA.DeviceCode;
using SiaNet.Backend.TensorSharp.Cpu;

namespace SiaNet
{
    public class ImgUtil
    {
        public static NDArray Im2Col(NDArray x, Tuple<uint, uint> kernalSize, int padding= 1, uint stride = 1, Tuple<uint, uint> dialation=null)
        {
            if (dialation == null)
                dialation = Tuple.Create<uint, uint>(1, 1);

            var (n, c, h, w) = x.GetConv2DShape();

            var out_height = (h + 2 * padding - kernalSize.Item1) / stride + 1;
            var out_width = (w + 2 * padding - kernalSize.Item2) / stride + 1;
            NDArray cols = new NDArray(DeviceManager.Current, DType.Float32, (c * kernalSize.Item1 * kernalSize.Item2), (n * out_height * out_width));
            if (DeviceManager.IsCuda)
            {

                Im2ColCuda im2ColKernels = new Im2ColCuda();
                im2ColKernels.Im2Col(x, cols, (int)c, (int)h, (int)w, (int)kernalSize.Item1, (int)kernalSize.Item2, 
                                    padding, padding, (int)stride, (int)stride, (int)dialation.Item1, (int)dialation.Item2);
            }
            else
            {
                Im2ColCpu im2ColKernels = new Im2ColCpu();
                im2ColKernels.Im2Cols(x, cols, (int)c, (int)h, (int)w, (int)kernalSize.Item1, (int)kernalSize.Item2,
                                    padding, padding, (int)stride, (int)stride, (int)dialation.Item1, (int)dialation.Item2);
            }

            return cols;
        }

        public static NDArray Col2Im(NDArray cols, long[] x_shape, Tuple<uint, uint> kernalSize, int padding = 1, uint stride = 1, Tuple<uint, uint> dialation = null)
        {

            if (dialation == null)
                dialation = Tuple.Create<uint, uint>(1, 1);

            NDArray im = new NDArray(DeviceManager.Current, DType.Float32, x_shape);

            if (DeviceManager.IsCuda)
            {
                Im2ColCuda im2ColKernels = new Im2ColCuda();
                im2ColKernels.Col2Im(cols, im, (int)x_shape[1], (int)x_shape[2], (int)x_shape[3], (int)kernalSize.Item1, (int)kernalSize.Item2
                            , padding, padding, (int)stride, (int)stride, (int)dialation.Item1, (int)dialation.Item2);
            }
            else
            {
                Im2ColCpu im2ColKernels = new Im2ColCpu();
                im2ColKernels.Cols2Im(cols, im, (int)x_shape[1], (int)x_shape[2], (int)x_shape[3], (int)kernalSize.Item1, (int)kernalSize.Item2
                           , padding, padding, (int)stride, (int)stride, (int)dialation.Item1, (int)dialation.Item2);
            }


            return im;
        }
    }

}
