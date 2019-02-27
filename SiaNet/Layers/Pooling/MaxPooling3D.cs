using SiaNet.Engine;
using System;
using System.Collections.Generic;
using System.Text;

namespace SiaNet.Layers
{
    public class MaxPooling3D : BaseLayer
    {
        public Tuple<int, int, int> PoolSize { get; set; }

        public int Strides { get; set; }

        public PaddingType Padding { get; set; }

        private Tensor xCols;

        public MaxPooling3D(Tuple<int, int, int> poolSize = null, int strides = 1, PaddingType padding = PaddingType.Same)
            : base("maxpooling3d")
        {
            PoolSize = poolSize ?? Tuple.Create<int, int, int>(2, 2, 2);
            Strides = strides;
            Padding = padding;
        }

        public override void Forward(Tensor x)
        {
            base.Forward(x);
            var (n, c, d, h, w) = x.GetConv3DShape();

            int pad = 0;
            if (Padding == PaddingType.Same)
            {
                pad = 1;
            }
            else if (Padding == PaddingType.Full)
            {
                pad = 2;
            }

            var d_out = (d - PoolSize.Item1) / Strides + 1;
            var h_out = (h - PoolSize.Item2) / Strides + 1;
            var w_out = (w - PoolSize.Item3) / Strides + 1;

            var x_reshaped = x.Reshape(n * c, 1, d, h, w);
            //xCols = K.Im2Col(x_reshaped, PoolSize, pad, Strides);
            Output = K.Argmax(xCols, 0);
            Output = Output.Reshape(d_out, h_out, w_out, n, c).Transpose(2, 3, 4, 0, 1);
        }

        public override void Backward(Tensor outputgrad)
        {
            Tensor dX_col = K.Constant(0, xCols.Shape);
            var (n, c, d, h, w) = Input.Data.GetConv3DShape();
            
            int pad = 0;
            if (Padding == PaddingType.Same)
            {
                pad = 1;
            }
            else if (Padding == PaddingType.Full)
            {
                pad = 2;
            }

            var dout_flat = outputgrad.Transpose(2, 3, 4, 0, 1).Reshape(1, -1);
            //var dX = K.Col2Im(dout_flat, Input.Data.Shape, PoolSize, pad, Strides);
            //Input.Grad = dX.Reshape(n, c, d, h, w);
        }
    }
}
