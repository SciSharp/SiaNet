using System;
using System.Collections.Generic;
using System.Text;
using SiaNet.Constraints;
using SiaNet.Initializers;
using SiaNet.Layers.Activations;
using SiaNet.Regularizers;
using TensorSharp;
using TensorSharp.Expression;

namespace SiaNet.Layers
{
    public class Dense : BaseLayer
    {
        public int Dim { get; set; }

        public BaseLayer Activation { get; set; }

        public bool UseBias { get; set; }

        public BaseInitializer KernalInitializer { get; set; }

        public BaseInitializer BiasInitializer { get; set; }

        public BaseConstraint KernalConstraint { get; set; }

        public BaseConstraint BiasConstraint { get; set; }

        public BaseRegularizer KernalRegularizer { get; set; }

        public BaseRegularizer BiasRegularizer { get; set; }

        public Dense(int dim, ActivationType activation = ActivationType.Linear,
                    BaseInitializer kernalInitializer = null, BaseRegularizer kernalRegularizer = null, BaseConstraint kernalConstraint = null,
                    bool useBias = false, BaseInitializer biasInitializer = null, BaseRegularizer biasRegularizer = null, BaseConstraint biasConstraint = null)
            : base("dense")
        {
            Dim = dim;
            Activation = ActivationRegistry.Get(activation);
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
            Variable weight = BuildVar("w", new long[] { x.Data.Shape[1], Dim }, x.Data.ElementType, KernalInitializer, KernalConstraint, KernalRegularizer);
            Variable bias = null;
            Output = Dot(x.Data, weight.Data);

            if (UseBias)
            {
                bias = BuildVar("b", new long[] { 1, Dim }, x.Data.ElementType, BiasInitializer, BiasConstraint, BiasRegularizer);
                Output += bias.Data;
            }

            if (Activation!=null)
                Activation.Forward(Output.ToVariable());
        }

        public override void Backward(Tensor outputgrad)
        {
            if (Activation != null)
                Activation.Backward(outputgrad);

            Input.Grad = Dot(outputgrad, Params["w"].Data.Transpose());
            Params["w"].Grad = Dot(Input.Data.Transpose(), outputgrad);
            if (UseBias)
                Params["b"].Grad = Sum(outputgrad, 0);
        }
    }
}
