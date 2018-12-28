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

        public Tuple<uint, uint> Strides { get; set; }

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

        public Conv2D(uint filters, Tuple<uint, uint> kernalSize, Tuple<uint, uint> strides = null, uint? padding = null, Tuple<uint, uint> dialationRate = null,
                    ActivationType activation = ActivationType.Linear, BaseInitializer kernalInitializer = null, BaseRegularizer kernalRegularizer = null,
                    BaseConstraint kernalConstraint = null, bool useBias = true, BaseInitializer biasInitializer = null, BaseRegularizer biasRegularizer = null,
                    BaseConstraint biasConstraint = null)
            : base("conv2d")
        {
            Filters = filters;
            KernalSize = kernalSize;
            Strides = strides ?? Tuple.Create<uint, uint>(1, 1);
            Padding = padding;
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
        public override void Forward(Variable x)
        {
            
            throw new NotImplementedException();
        }

        public override void Backward(Tensor outputgrad)
        {
            throw new NotImplementedException();
        }
    }
}
