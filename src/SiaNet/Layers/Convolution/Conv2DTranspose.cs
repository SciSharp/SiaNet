using SiaDNN.Constraints;
using SiaDNN.Initializers;
using SiaNet.Backend;
using SiaNet.Regularizers;
using System;
using System.Collections.Generic;
using System.Text;

namespace SiaNet.Layers
{
    public class Conv2DTranspose : BaseLayer, ILayer
    {
        public uint Filters { get; set; }

        public Tuple<uint, uint> KernalSize { get; set; }

        public Tuple<uint, uint> Strides { get; set; }

        public uint? Padding { get; set; }

        public DeconvolutionLayout DataFormat { get; set; }

        public ActivationType Activation { get; set; }

        public Tuple<uint, uint> DialationRate { get; set; }

        public bool UseBias { get; set; }

        public BaseInitializer BiasInitializer { get; set; }

        public BaseInitializer KernalInitializer { get; set; }

        public BaseConstraint KernalConstraint { get; set; }

        public BaseConstraint BiasConstraint { get; set; }

        public BaseRegularizer KernalRegularizer { get; set; }

        public BaseRegularizer BiasRegularizer { get; set; }

        public Conv2DTranspose(uint filters, Tuple<uint, uint> kernalSize, Tuple<uint, uint> strides = null, uint? padding = null,
                                DeconvolutionLayout dataFormat = DeconvolutionLayout.None, Tuple<uint, uint> dialationRate = null, 
                                ActivationType activation = ActivationType.Linear, BaseInitializer kernalInitializer = null, BaseRegularizer kernalRegularizer = null,
                                BaseConstraint kernalConstraint = null, bool useBias = true, BaseInitializer biasInitializer =null, BaseRegularizer biasRegularizer = null,
                                BaseConstraint biasConstraint = null)
            : base("conv2dtranspose")
        {
            Filters = filters;
            KernalSize = kernalSize ?? Tuple.Create<uint, uint>(1, 1);
            Strides = strides ?? Tuple.Create<uint, uint>(1, 1);
            Padding = padding;
            DataFormat = dataFormat;
            DialationRate = dialationRate ?? Tuple.Create<uint, uint>(1, 1);
            Activation = activation;
            UseBias = useBias;
            KernalInitializer = kernalInitializer ?? new GlorotUniform();
            BiasInitializer = biasInitializer ?? new Zeros();
            KernalConstraint = kernalConstraint;
            BiasConstraint = biasConstraint;
            KernalRegularizer = kernalRegularizer;
            BiasRegularizer = biasRegularizer;
        }

        public Symbol Build(Symbol x)
        {
            var biasName = UUID.GetID(ID + "_b");
            var weightName = UUID.GetID(ID + "_w");
            Shape pad = null;
            if (Padding.HasValue)
            {
                pad = new Shape(Padding.Value, Padding.Value);
            }
            else
            {
                pad = new Shape();
            }

            InitParams.Add(biasName, BiasInitializer);
            InitParams.Add(weightName, KernalInitializer);

            ConstraintParams.Add(weightName, KernalConstraint);
            ConstraintParams.Add(biasName, BiasConstraint);

            RegularizerParams.Add(weightName, KernalRegularizer);
            RegularizerParams.Add(biasName, BiasRegularizer);

            return Operators.Deconvolution(ID, x, Symbol.Variable(weightName), Symbol.Variable(biasName), new Shape(KernalSize.Item1, KernalSize.Item2),
                                    Filters, new Shape(Strides.Item1, Strides.Item2), new Shape(DialationRate.Item1, DialationRate.Item2), pad,
                                    new Shape(), new Shape(), 1, 512, !UseBias, DeconvolutionCudnnTune.None, GlobalParam.UseCudnn, DataFormat);
        }
    }
}
