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
    public class Conv2D : BaseLayer
    {
        public uint Filters { get; set; }

        public Tuple<uint, uint> KernalSize { get; set; }

        public uint Strides { get; set; }

        public uint? Padding { get; set; }

        public Tuple<uint, uint> DialationRate { get; set; }

        public ActivationType Activation { get; set; }

        public bool UseBias { get; set; }

        public BaseInitializer BiasInitializer { get; set; }

        public BaseInitializer KernalInitializer { get; set; }

        public BaseConstraint KernalConstraint { get; set; }

        public BaseConstraint BiasConstraint { get; set; }

        public BaseRegularizer KernalRegularizer { get; set; }

        public BaseRegularizer BiasRegularizer { get; set; }

        private Tensor xCols;

        public Conv2D(uint filters, Tuple<uint, uint> kernalSize, uint strides = 1, uint? padding = null, Tuple<uint, uint> dialationRate = null, 
                ActivationType activation = ActivationType.Linear, BaseInitializer kernalInitializer = null,
                        BaseRegularizer kernalRegularizer = null, BaseConstraint kernalConstraint = null, bool useBias = true,
                        BaseInitializer biasInitializer = null, BaseRegularizer biasRegularizer = null, BaseConstraint biasConstraint = null)
            : base("conv2d")
        {
            Filters = filters;
            KernalSize = kernalSize;
            Strides = strides;
            Padding = padding;
            DialationRate = dialationRate ?? new Tuple<uint, uint>(1, 1);
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
            var (n, c, h, w) = x.Data.GetConv2DShape();

            Variable weight = BuildVar("w", new long[] { Filters, c, KernalSize.Item1, KernalSize.Item2 }, x.Data.ElementType, KernalInitializer, KernalConstraint, KernalRegularizer);
            Variable bias = null;
            if (UseBias)
            {
                bias = BuildVar("b", new long[] { Filters, 1}, x.Data.ElementType, BiasInitializer, BiasConstraint, BiasRegularizer);
            }

            var h_out = (h - KernalSize.Item1 + 2 * Padding) / Strides + 1;
            var w_out = (w - KernalSize.Item2 + 2 * Padding) / Strides + 1;

            xCols = ImgUtil.Im2Col(x.Data, KernalSize.Item1, KernalSize.Item2, Padding, Strides);
            var wRows = weight.Data.Reshape(Filters, -1);

            Output = Dot(wRows, xCols);
            if(UseBias)
            {
                Output = Output + bias.Data;
            }

            Output = Output.Reshape(Filters, h_out.Value, w_out.Value, n).Transpose(3, 0, 1, 2);
        }

        public override void Backward(Tensor outputgrad)
        {
            var dout_flat = outputgrad.Transpose(3, 0, 1, 2).Reshape(Filters, -1);
            var dW = Dot(dout_flat, xCols.Transpose());
            dW = dW.Reshape(Params["w"].Data.Shape);
            var db = Sum(outputgrad, 0, 2, 3).Reshape(Filters, -1);
            var W_flat = Params["w"].Data.Reshape(Filters, -1);

            var dX_col = Dot(W_flat.Transpose(), dout_flat);
            Input.Grad = ImgUtil.Col2Im(dX_col, Input.Data.Shape, KernalSize.Item1, KernalSize.Item2, Padding, Strides);

            Params["w"].Grad = dW;
            if(UseBias)
                Params["b"].Grad = db;
        }
    }
}
