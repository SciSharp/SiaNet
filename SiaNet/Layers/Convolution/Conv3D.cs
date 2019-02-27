using SiaNet.Constraints;
using SiaNet.Engine;
using SiaNet.Initializers;
using SiaNet.Regularizers;
using System;
using System.Collections.Generic;
using System.Text;

namespace SiaNet.Layers
{
    public class Conv3D : BaseLayer
    {
        public int Filters { get; set; }

        public Tuple<int, int, int> KernalSize { get; set; }

        public int Strides { get; set; }

        public PaddingType Padding { get; set; }

        public Tuple<int, int, int> DialationRate { get; set; }

        public ActType Act { get; set; }

        public bool UseBias { get; set; }

        public BaseInitializer BiasInitializer { get; set; }

        public BaseInitializer KernalInitializer { get; set; }

        public BaseConstraint KernalConstraint { get; set; }

        public BaseConstraint BiasConstraint { get; set; }

        public BaseRegularizer KernalRegularizer { get; set; }

        public BaseRegularizer BiasRegularizer { get; set; }

        private Tensor xCols;

        public Conv3D(int filters, Tuple<int, int, int> kernalSize, int strides = 1, PaddingType padding = PaddingType.Same, Tuple<int, int, int> dialationRate = null,
                        ActType activation = ActType.Linear, BaseInitializer kernalInitializer = null, BaseRegularizer kernalRegularizer = null,
                        BaseConstraint kernalConstraint = null, bool useBias = true, BaseInitializer biasInitializer = null, BaseRegularizer biasRegularizer = null, BaseConstraint biasConstraint = null)
            : base("conv3d")
        {
            Filters = filters;
            KernalSize = kernalSize;
            Strides = strides;
            Padding = padding;
            DialationRate = dialationRate ?? Tuple.Create(1, 1, 1);
            Act = activation;
            UseBias = useBias;
            KernalInitializer = kernalInitializer ?? new GlorotUniform();
            BiasInitializer = biasInitializer ?? new Zeros();
            KernalConstraint = kernalConstraint;
            BiasConstraint = biasConstraint;
            KernalRegularizer = kernalRegularizer;
            BiasRegularizer = biasRegularizer;
        }

        public override void Forward(Tensor x)
        {
            base.Forward(x);
            var (n, c, d, h, w) = x.GetConv3DShape();

            Parameter weight = BuildParam("w", new long[] { Filters, c, KernalSize.Item1, KernalSize.Item2, KernalSize.Item2 }, x.ElementType, KernalInitializer, KernalConstraint, KernalRegularizer);
            Parameter bias = null;
            if (UseBias)
            {
                bias = BuildParam("b", new long[] { Filters, 1 }, x.ElementType, BiasInitializer, BiasConstraint, BiasRegularizer);
            }

            int pad = 0;
            if (Padding == PaddingType.Same)
            {
                pad = 1;
            }
            else if (Padding == PaddingType.Full)
            {
                pad = 2;
            }

            var d_out = (d - KernalSize.Item1 + 2 * pad) / Strides + 1;
            var h_out = (h - KernalSize.Item2 + 2 * pad) / Strides + 1;
            var w_out = (w - KernalSize.Item3 + 2 * pad) / Strides + 1;

            //xCols = K.Im2Col(x, KernalSize, pad, Strides);
            var wRows = weight.Data.Reshape(Filters, -1);

            Output = K.Dot(wRows,xCols);
            if (UseBias)
            {
                Output = Output + bias.Data;
            }

            Output = Output.Reshape(Filters, d_out, h_out, w_out, n).Transpose(4, 0, 1, 2, 3);
        }

        public override void Backward(Tensor outputgrad)
        {
            int pad = 0;
            if (Padding == PaddingType.Same)
            {
                pad = 1;
            }
            else if (Padding == PaddingType.Full)
            {
                pad = 2;
            }

            var dout_flat = outputgrad.Transpose(4, 0, 1, 2, 3).Reshape(Filters, -1);
            var dW = K.Dot(dout_flat, xCols.Transpose());
            dW = dW.Reshape(Params["w"].Data.Shape);
            var db = K.Sum(outputgrad, 0, 2, 3, 4).Reshape(Filters, -1);
            var W_flat = Params["w"].Data.Reshape(Filters, -1);

            var dX_col = K.Dot(W_flat.Transpose(), dout_flat);
            //Input.Grad = K.Col2Im(dX_col, Input.Data.Shape, KernalSize, pad, Strides);

            Params["w"].Grad = dW;
            if (UseBias)
                Params["b"].Grad = db;
        }
    }
}
