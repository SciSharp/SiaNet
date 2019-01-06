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
            if (padding.HasValue)
            {
                x = x.Pad(1, padding.Value);
            }

            var list = x.NDto2DList();
            var cols = x.Transpose(1, 2, 3, 4, 0).Unfold(3, kernalSize.Item3, stride).Unfold(2, kernalSize.Item2, stride).Unfold(1, kernalSize.Item1, stride);

            return cols;
        }

        public static Tensor Im2Col(Tensor x, Tuple<uint, uint> kernalSize, uint? padding=null, uint stride = 1)
        {
            var (n, c, h, w) = x.GetConv2DShape();
            if (padding.HasValue)
            {
                x = x.Pad(1, padding.Value);
            }

            var cols = x.Transpose(0, 1, 2, 3).Unfold(2, kernalSize.Item2, stride).Unfold(3, kernalSize.Item1, stride);
            var x_t = x.Transpose(1, 0, 2, 3);
            x_t.Narrow(2, 0, 3).Narrow(3, 0, 3).Print();

            return cols;
        }

        public static Tensor Im2Col(Tensor x, uint steps, uint? padding = null, uint stride = 1)
        {

            if (padding.HasValue)
            {
                x = x.Pad(1, padding.Value);
            }

            x.Print();

            var cols = x.Transpose(2,1,0).Unfold(1, 1, stride).Unfold(0, 3, 1);

            return cols;
        }

        public static Tensor Col2Im(Tensor cols, long[] x_shape, uint field_height, uint field_width, uint field_depth, uint? padding = null, uint stride = 1)
        {
            return cols;
        }

        public static Tensor Col2Im(Tensor cols, long[] x_shape, uint field_height, uint field_width, uint? padding = null, uint stride = 1)
        {
            return cols;
        }

        public static Tensor Col2Im(Tensor cols, long[] x_shape, uint field_height, uint? padding = null, uint stride = 1)
        {
            return cols;
        }
    }
}
