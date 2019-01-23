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

        public override void Forward(Tensor x)
        {
            Input = x.ToParameter();
            
            Parameter weight = BuildParam("w", new long[] { x.Shape[1], Dim }, x.ElementType, KernalInitializer, KernalConstraint, KernalRegularizer);
            Parameter bias = null;
            Output = Dot(x, weight.Data);

            if (UseBias)
            {
                bias = BuildParam("b", new long[] { 1, Dim }, x.ElementType, BiasInitializer, BiasConstraint, BiasRegularizer);
                Output += bias.Data;
            }

            if (Activation != null)
            {
                Activation.Forward(Output);
                Output = Activation.Output;
            }
        }

        public override void Backward(Tensor outputgrad)
        {
            if (Activation != null)
            {
                Activation.Backward(outputgrad);
                outputgrad = Activation.Input.Grad;
            }

            Input.Grad = Dot(outputgrad, base["w"].Data.Transpose());
            base["w"].Grad = Dot(Input.Data.Transpose(), outputgrad);
            if (UseBias)
                base["b"].Grad = Sum(outputgrad, 0);
        }
    }
}
