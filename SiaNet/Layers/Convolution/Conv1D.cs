using SiaNet.Constraints;
using SiaNet.Engine;
using SiaNet.Initializers;
using SiaNet.Regularizers;
using System;
using System.Collections.Generic;
using System.Text;

namespace SiaNet.Layers
{
    public class Conv1D : BaseLayer
    {
        public int Filters { get; set; }

        public int KernalSize { get; set; }

        public int Strides { get; set; }

        public PaddingType Padding { get; set; }

        public int DilationRate { get; set; }

        public ActType Act { get; set; }

        public bool UseBias { get; set; }

        public BaseInitializer BiasInitializer { get; set; }

        public BaseInitializer KernalInitializer { get; set; }

        public BaseConstraint KernalConstraint { get; set; }

        public BaseConstraint BiasConstraint { get; set; }

        public BaseRegularizer KernalRegularizer { get; set; }

        public BaseRegularizer BiasRegularizer { get; set; }

        private Tensor xCols;

        public Conv1D(int filters, int kernalSize, int strides = 1, PaddingType padding = PaddingType.Same, int dilationRate = 1,
                    ActType activation = ActType.Linear, BaseInitializer kernalInitializer = null, BaseRegularizer kernalRegularizer = null,
                    BaseConstraint kernalConstraint = null, bool useBias = true, BaseInitializer biasInitializer = null, BaseRegularizer biasRegularizer = null,
                    BaseConstraint biasConstraint = null)
            : base("conv1d")
        {
            Filters = filters;
            KernalSize = kernalSize;
            Strides = strides;
            Padding = padding;
            DilationRate = dilationRate;
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
            var (n, c, s) = x.GetConv1DShape();

            Parameter weight = BuildParam("w", new long[] { Filters, c, KernalSize }, x.ElementType, KernalInitializer, KernalConstraint, KernalRegularizer);
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

            KernalSize = (KernalSize - 1) * DilationRate + 1;

            var steps_out = (s - KernalSize + 2 * pad) / Strides + 1;

            xCols = K.Im2Col(x, Tuple.Create(KernalSize, KernalSize), pad, Strides);
            var wRows = weight.Data.Reshape(Filters, -1);
            
            Output = K.Dot(wRows, xCols);
            if (UseBias)
            {
                Output = Output + bias.Data;
            }

            Output = K.Transpose(Output.Reshape(Filters, steps_out, n), 2, 0, 1);
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

            var dout_flat = outputgrad.Transpose(2, 0, 1).Reshape(Filters, -1);
            var dW = K.Dot(dout_flat, xCols.Transpose());
            dW = dW.Reshape(Params["w"].Data.Shape);
            var db = K.Sum(outputgrad, 0, 1, 2).Reshape(Filters, -1);
            var W_flat = Params["w"].Data.Reshape(Filters, -1);

            var dX_col = K.Dot(W_flat.Transpose(), dout_flat);
            Input.Grad = K.Col2Im(dX_col, Input.Data.Shape, Tuple.Create(KernalSize, KernalSize), pad, Strides);

            Params["w"].Grad = dW;
            if (UseBias)
                Params["b"].Grad = db;
        }
    }
}
