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

        public Tuple<uint, uint> KernalSize { get; set; }

        public uint Strides { get; set; }

        public uint? Padding { get; set; }

        public uint DialationRate { get; set; }

        public ActivationType Activation { get; set; }

        public bool UseBias { get; set; }

        public BaseInitializer BiasInitializer { get; set; }

        public BaseInitializer KernalInitializer { get; set; }

        public BaseConstraint KernalConstraint { get; set; }

        public BaseConstraint BiasConstraint { get; set; }

        public BaseRegularizer KernalRegularizer { get; set; }

        public BaseRegularizer BiasRegularizer { get; set; }

        public Conv1D(uint filters, Tuple<uint, uint> kernalSize, uint strides = 1, uint? padding = null, uint dialationRate = 1, 
                ActivationType activation = ActivationType.Linear, BaseInitializer kernalInitializer = null,
                        BaseRegularizer kernalRegularizer = null, BaseConstraint kernalConstraint = null, bool useBias = true,
                        BaseInitializer biasInitializer = null, BaseRegularizer biasRegularizer = null, BaseConstraint biasConstraint = null)
            : base("conv1d")
        {
            Filters = filters;
            KernalSize = kernalSize;
            Strides = strides;
            Padding = padding;
            DialationRate = dialationRate;
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
            Input = x;
            Variable weight = BuildVar("w", new long[] { Filters, x.Data.Shape[1], KernalSize.Item1, KernalSize.Item2 }, x.Data.ElementType, KernalInitializer, KernalConstraint, KernalRegularizer);
            Variable bias = null;
            if (UseBias)
            {
                bias = BuildVar("b", new long[] { Filters, 1}, x.Data.ElementType, BiasInitializer, BiasConstraint, BiasRegularizer);
            }
        }

        public override void Backward(Tensor outputgrad)
        {
            throw new NotImplementedException();
        }
    }
}
