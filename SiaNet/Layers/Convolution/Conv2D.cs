using SiaNet.Constraints;
using SiaNet.Initializers;
using SiaNet.Regularizers;
using System;
using System.Collections.Generic;
using System.Text;
using System.Linq;
using SiaNet.Engine;

namespace SiaNet.Layers
{
    public class Conv2D : BaseLayer
    {
        public int Filters { get; set; }

        public Tuple<int, int> KernalSize { get; set; }

        public int Strides { get; set; }

        public PaddingType Padding { get; set; }

        public Tuple<int, int> DialationRate { get; set; }

        public ActType Act { get; set; }

        public bool UseBias { get; set; }

        public BaseInitializer BiasInitializer { get; set; }

        public BaseInitializer KernalInitializer { get; set; }

        public BaseConstraint KernalConstraint { get; set; }

        public BaseConstraint BiasConstraint { get; set; }

        public BaseRegularizer KernalRegularizer { get; set; }

        public BaseRegularizer BiasRegularizer { get; set; }

        private Tensor xCols;

        public Conv2D(int filters, Tuple<int, int> kernalSize, int strides = 1, PaddingType padding = PaddingType.Same, Tuple<int, int> dialationRate = null, 
                ActType activation = ActType.Linear, BaseInitializer kernalInitializer = null,
                        BaseRegularizer kernalRegularizer = null, BaseConstraint kernalConstraint = null, bool useBias = true,
                        BaseInitializer biasInitializer = null, BaseRegularizer biasRegularizer = null, BaseConstraint biasConstraint = null)
            : base("conv2d")
        {
            Filters = filters;
            KernalSize = kernalSize;
            Strides = strides;
            Padding = padding;
            DialationRate = dialationRate ?? Tuple.Create(1, 1);
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
            Input = x.ToParameter();
            var (n, c, h, w) = x.GetConv2DShape();

            Parameter weight = BuildParam("w", new long[] { Filters, c, KernalSize.Item1, KernalSize.Item2 }, x.ElementType, KernalInitializer, KernalConstraint, KernalRegularizer);
            Parameter bias = null;
            if (UseBias)
            {
                bias = BuildParam("b", new long[] { Filters, 1}, x.ElementType, BiasInitializer, BiasConstraint, BiasRegularizer);
            }

            int pad = 0;
            if(Padding == PaddingType.Same)
            {
                pad = 1;
            }
            else if(Padding == PaddingType.Full)
            {
                pad = 2;
            }

            var dialatedKernel = Tuple.Create(((KernalSize.Item1 - 1) * DialationRate.Item1 + 1), ((KernalSize.Item2 - 1) * DialationRate.Item2 + 1));

            var h_out = (h - dialatedKernel.Item1 + 2 * pad) / Strides + 1;
            var w_out = (w - dialatedKernel.Item2 + 2 * pad) / Strides + 1;
            
            var wRows = weight.Data.Reshape(Filters, -1);
            xCols = K.Im2Col(x, dialatedKernel, pad, Strides);
            Output = K.Dot(wRows, xCols);
            
            if(UseBias)
            {
                Output = Output + bias.Data;
            }

            Output = Output.Reshape(Filters, h_out, w_out, n);
            Output = Output.Transpose(3, 0, 1, 2);
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

            var dout_flat = outputgrad.Transpose(1, 2, 3, 0).Reshape(Filters, -1);
            var dW = K.Dot(dout_flat, xCols.Transpose());
            dW = dW.Reshape(base["w"].Data.Shape);

            var W_flat = base["w"].Data.Reshape(Filters, -1);
            var dX_col = K.Dot(W_flat.Transpose(), dout_flat);
            Input.Grad = K.Col2Im(dX_col, Input.Data.Shape, KernalSize, pad, Strides);

            this["w"].Grad = dW;
            
            if (UseBias)
            {
                var db = K.Sum(outputgrad, 0, 2, 3).Reshape(Filters, -1);
                this["b"].Grad = db;
            }
        }
    }
}
