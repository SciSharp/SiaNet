using SiaNet.Constraints;
using SiaNet.Initializers;
using SiaNet.Regularizers;
using System;
using System.Collections.Generic;
using System.Text;
using TensorSharp;
using TensorSharp.Cpu;

namespace SiaNet.Layers
{
    public class Conv1D : BaseLayer
    {
        public uint Filters { get; set; }

        public uint KernalSize { get; set; }

        public uint Strides { get; set; }

        public PaddingType Padding { get; set; }

        public uint DilationRate { get; set; }

        public ActivationType Activation { get; set; }

        public bool UseBias { get; set; }

        public BaseInitializer BiasInitializer { get; set; }

        public BaseInitializer KernalInitializer { get; set; }

        public BaseConstraint KernalConstraint { get; set; }

        public BaseConstraint BiasConstraint { get; set; }

        public BaseRegularizer KernalRegularizer { get; set; }

        public BaseRegularizer BiasRegularizer { get; set; }

        private Tensor xCols;

        public Conv1D(uint filters, uint kernalSize, uint strides = 1, PaddingType padding = PaddingType.Same, uint dilationRate = 1,
                    ActivationType activation = ActivationType.Linear, BaseInitializer kernalInitializer = null, BaseRegularizer kernalRegularizer = null,
                    BaseConstraint kernalConstraint = null, bool useBias = true, BaseInitializer biasInitializer = null, BaseRegularizer biasRegularizer = null,
                    BaseConstraint biasConstraint = null)
            : base("conv1d")
        {
            Filters = filters;
            KernalSize = kernalSize;
            Strides = strides;
            Padding = padding;
            DilationRate = dilationRate;
            Activation = activation;
            UseBias = useBias;
            KernalInitializer = kernalInitializer ?? new GlorotUniform();
            BiasInitializer = biasInitializer ?? new Zeros();
            KernalConstraint = kernalConstraint;
            BiasConstraint = biasConstraint;
            KernalRegularizer = kernalRegularizer;
            BiasRegularizer = biasRegularizer;
        }
        public override void Forward(Variable x)
        {
            //ToDo: Implement DilationRate
            Input = x;
            var (n, c, s) = x.Data.GetConv1DShape();

            Variable weight = BuildVar("w", new long[] { Filters, c, KernalSize }, x.Data.ElementType, KernalInitializer, KernalConstraint, KernalRegularizer);
            Variable bias = null;
            if (UseBias)
            {
                bias = BuildVar("b", new long[] { Filters, 1 }, x.Data.ElementType, BiasInitializer, BiasConstraint, BiasRegularizer);
            }

            uint? pad = null;
            if (Padding == PaddingType.Same)
            {
                pad = 1;
            }
            else if (Padding == PaddingType.Full)
            {
                pad = 2;
            }

            var steps_out = (s - KernalSize + 2 * pad) / Strides + 1;

            xCols = ImgUtil.Im2Col(x.Data, KernalSize, pad, Strides);
            var wRows = weight.Data.Reshape(Filters, -1);

            Output = Dot(wRows, xCols);
            if (UseBias)
            {
                Output = Output + bias.Data;
            }

            Output = Output.Reshape(Filters, steps_out.Value, n).Transpose(2, 0, 1);
        }

        public override void Backward(Tensor outputgrad)
        {
            uint? pad = null;
            if (Padding == PaddingType.Same)
            {
                pad = 1;
            }
            else if (Padding == PaddingType.Full)
            {
                pad = 2;
            }

            var dout_flat = outputgrad.Transpose(2, 0, 1).Reshape(Filters, -1);
            var dW = Dot(dout_flat, xCols.Transpose());
            dW = dW.Reshape(Params["w"].Data.Shape);
            var db = Sum(outputgrad, 0, 1, 2).Reshape(Filters, -1);
            var W_flat = Params["w"].Data.Reshape(Filters, -1);

            var dX_col = Dot(W_flat.Transpose(), dout_flat);
            Input.Grad = ImgUtil.Col2Im(dX_col, Input.Data.Shape, KernalSize, pad, Strides);

            Params["w"].Grad = dW;
            if (UseBias)
                Params["b"].Grad = db;
        }
    }
}
