using System;
using System.Collections.Generic;
using System.Text;
using TensorSharp;
using System.Linq;
using TensorSharp.CUDA.DeviceCode;
using TensorSharp.Cpu;

namespace SiaNet
{
    public class ImgUtil
    {
        public static Tensor Im2Col(Tensor x, Tuple<uint, uint, uint> kernalSize, uint? padding = null, uint stride = 1)
        {
            var (n, c, d, h, w) = x.GetConv3DShape();
            if (padding.HasValue)
            {
                x = x.Pad(1, padding.Value);
            }

            var list = x.NDto2DList();
            var cols = x.Transpose(1, 2, 3, 4, 0).Unfold(3, kernalSize.Item3, stride).Unfold(2, kernalSize.Item2, stride).Unfold(1, kernalSize.Item1, stride);

            return Ops.NewContiguous(cols).Reshape(c * d * h * w, -1);
        }

        public static Tensor Im2Col(Tensor x, Tuple<uint, uint> kernalSize, uint? padding=null, uint stride = 1, Tuple<uint, uint> dialation=null)
        {
            if (dialation == null)
                dialation = Tuple.Create<uint, uint>(0, 0);

            if (!padding.HasValue)
                padding = 0;

            var (n, c, h, w) = x.GetConv2DShape();

            var out_height = (h + 2 * padding.Value - kernalSize.Item1) / stride + 1;
            var out_width = (w + 2 * padding.Value - kernalSize.Item2) / stride + 1;
            Tensor cols = new Tensor(Global.Device, DType.Float32, (c * kernalSize.Item1 * kernalSize.Item2), (n * out_height * out_width));
            if (Global.UseCuda)
            {

                Im2ColCuda im2ColKernels = new Im2ColCuda();
                im2ColKernels.Im2Col(x, cols, (int)c, (int)h, (int)w, (int)kernalSize.Item1, (int)kernalSize.Item2, 
                                    (int)padding.Value, (int)padding.Value, (int)stride, (int)stride, (int)dialation.Item1, (int)dialation.Item2);
            }
            else
            {
                Im2ColCpu im2ColKernels = new Im2ColCpu();
                im2ColKernels.Im2Cols(x, cols, (int)c, (int)h, (int)w, (int)kernalSize.Item1, (int)kernalSize.Item2,
                                    (int)padding.Value, (int)padding.Value, (int)stride, (int)stride, (int)dialation.Item1, (int)dialation.Item2);
            }

            return cols.Reshape(c * kernalSize.Item1 * kernalSize.Item2, -1);
        }

        public static Tensor Im2Col(Tensor x, uint steps, uint? padding = null, uint stride = 1)
        {
            var (n, c, s) = x.GetConv1DShape();
            if (padding.HasValue)
            {
                x = x.Pad(1, padding.Value);
            }

            var cols = x.Transpose(2, 1, 0).Unfold(1, 1, stride).Unfold(0, steps, stride);
            cols.Print();
            return Ops.NewContiguous(cols).Reshape(c * s, -1);
        }

        public static Tensor Col2Im(Tensor cols, long[] x_shape, Tuple<uint, uint, uint> kernalSize, uint? padding = null, uint stride = 1)
        {
            return cols;
        }

        public static Tensor Col2Im(Tensor cols, long[] x_shape, Tuple<uint, uint> kernalSize, uint? padding = null, uint stride = 1, Tuple<uint, uint> dialation = null)
        {
            
            if (dialation == null)
                dialation = Tuple.Create<uint, uint>(0, 0);

            if (!padding.HasValue)
                padding = 0;

            Tensor im = new Tensor(Global.Device, DType.Float32, x_shape);

            if(Global.UseCuda)
            {
                Im2ColCuda im2ColKernels = new Im2ColCuda();
                im2ColKernels.Col2Im(cols, im, (int)x_shape[1], (int)x_shape[2], (int)x_shape[3], (int)kernalSize.Item1, (int)kernalSize.Item2
                            , (int)padding.Value, (int)padding.Value, (int)stride, (int)stride, (int)dialation.Item1, (int)dialation.Item2);
            }
            else
            {
                Im2ColCpu im2ColKernels = new Im2ColCpu();
                im2ColKernels.Cols2Im(cols, im, (int)x_shape[1], (int)x_shape[2], (int)x_shape[3], (int)kernalSize.Item1, (int)kernalSize.Item2
                           , (int)padding.Value, (int)padding.Value, (int)stride, (int)stride, (int)dialation.Item1, (int)dialation.Item2);
            }
            

            return im;
        }

        public static Tensor Col2Im(Tensor cols, long[] x_shape, uint steps, uint? padding = null, uint stride = 1)
        {
            return cols;
        }
    }

}
