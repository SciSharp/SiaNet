using SiaNet.Engine;
using System;
using System.Collections.Generic;
using System.Text;

namespace SiaNet.Layers
{
    public class AvgPooling2D : BaseLayer
    {
        public Tuple<int, int> PoolSize { get; set; }

        public int Strides { get; set; }

        public PaddingType Padding { get; set; }

        private Tensor xCols;

        public AvgPooling2D(Tuple<int, int> poolSize = null, int strides = 1, PaddingType padding = PaddingType.Same)
            : base("avgpooling2d")
        {
            PoolSize = poolSize ?? Tuple.Create(2, 2);
            Strides = strides;
            Padding = padding;
        }

        public override void Forward(Tensor x)
        {
            base.Forward(x);
            var (n, c, h, w) = x.GetConv2DShape();

            int pad = 0;
            if (Padding == PaddingType.Same)
            {
                pad = 1;
            }
            else if (Padding == PaddingType.Full)
            {
                pad = 2;
            }

            var h_out = (h - PoolSize.Item1) / Strides + 1;
            var w_out = (w - PoolSize.Item2) / Strides + 1;

            var x_reshaped = x.Reshape(n * c, 1, h, w);
            xCols = K.Im2Col(x_reshaped, PoolSize, pad, Strides);
            Output = K.Mean(xCols, 0);
            Output = Output.Reshape(h_out, w_out, n, c).Transpose(2, 3, 0, 1);
        }

        public override void Backward(Tensor outputgrad)
        {
            Tensor dX_col = K.Constant(0, xCols.Shape);
            var (n, c, h, w) = Input.Data.GetConv2DShape();
            
            int pad = 0;
            if (Padding == PaddingType.Same)
            {
                pad = 1;
            }
            else if (Padding == PaddingType.Full)
            {
                pad = 2;
            }

            var dout_flat = outputgrad.Transpose(2, 3, 0, 1).Reshape(1, -1);
            var dX = K.Col2Im(dout_flat, Input.Data.Shape, PoolSize, pad, Strides);
            Input.Grad = dX.Reshape(n, c, h, w);
        }
    }
}
