using System;
using System.Collections.Generic;
using System.Text;
using TensorSharp;
using System.Linq;

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

        public static Tensor Im2Col(Tensor x, Tuple<uint, uint> kernalSize, uint? padding=null, uint stride = 1)
        {
            var (n, c, h, w) = x.GetConv2DShape();
            if (padding.HasValue)
            {
                x = x.Pad(1, padding.Value);
            }

            var cols = x.Transpose(0, 1, 2, 3).Unfold(2, kernalSize.Item2, stride).Unfold(3, kernalSize.Item1, stride);
            return Ops.NewContiguous(cols).Reshape(c * h * w, -1);
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

        public static Tensor Col2Im(Tensor cols, long[] x_shape, Tuple<uint, uint> kernalSize, uint? padding = null, uint stride = 1)
        {
            return cols;
        }

        public static Tensor Col2Im(Tensor cols, long[] x_shape, uint steps, uint? padding = null, uint stride = 1)
        {
            return cols;
        }
    }
}
