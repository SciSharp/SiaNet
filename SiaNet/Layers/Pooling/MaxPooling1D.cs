using System;
using System.Collections.Generic;
using System.Text;
using TensorSharp;

namespace SiaNet.Layers
{
    public class MaxPooling1D : BaseLayer
    {
        public uint PoolSize { get; set; }

        public uint Strides { get; set; }

        public PaddingType Padding { get; set; }

        private Tensor xCols;

        public MaxPooling1D(uint poolSize = 2, uint strides = 1, PaddingType padding = PaddingType.Same)
            : base("maxpooling1d")
        {
            PoolSize = poolSize;
            Strides = strides;
            Padding = padding;
        }

        public override void Forward(Parameter x)
        {
            Input = x;
            var (n, c, s) = x.Data.GetConv1DShape();

            uint? pad = null;
            if (Padding == PaddingType.Same)
            {
                pad = 1;
            }
            else if (Padding == PaddingType.Full)
            {
                pad = 2;
            }

            var s_out = (s - PoolSize) / Strides + 1;

            var x_reshaped = x.Data.Reshape(n * c, 1, s);
            xCols = ImgUtil.Im2Col(x_reshaped, PoolSize, pad, Strides);
            Output = Argmax(xCols, 0);
            Output = Output.Reshape(s_out, n, c).Transpose(2, 0, 1);
        }

        public override void Backward(Tensor outputgrad)
        {
            Tensor dX_col = new Tensor(xCols.Allocator, xCols.ElementType, xCols.Shape);
            var (n, c, s) = Input.Data.GetConv1DShape();

            Fill(dX_col, 0);
            uint? pad = null;
            if (Padding == PaddingType.Same)
            {
                pad = 1;
            }
            else if (Padding == PaddingType.Full)
            {
                pad = 2;
            }

            var dout_flat = outputgrad.Transpose(2, 0, 1).Reshape(1, -1);
            var dX = ImgUtil.Col2Im(dout_flat, Input.Data.Shape, PoolSize, pad, Strides);
            Input.Grad = dX.Reshape(n, c, s);
        }
    }
}
