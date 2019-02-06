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

        public BaseLayer Act { get; set; }

        public bool UseBias { get; set; }

        public BaseInitializer KernalInitializer { get; set; }

        public BaseInitializer BiasInitializer { get; set; }

        public BaseConstraint KernalConstraint { get; set; }

        public BaseConstraint BiasConstraint { get; set; }

        public BaseRegularizer KernalRegularizer { get; set; }

        public BaseRegularizer BiasRegularizer { get; set; }

        public Dense(int dim, ActType activation = ActType.Linear,
                    BaseInitializer kernalInitializer = null, BaseRegularizer kernalRegularizer = null, BaseConstraint kernalConstraint = null,
                    bool useBias = true, BaseInitializer biasInitializer = null, BaseRegularizer biasRegularizer = null, BaseConstraint biasConstraint = null)
            : base("dense")
        {
            Dim = dim;
            Act = ActivationRegistry.Get(activation);
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

            if (Act != null)
            {
                Act.Forward(Output);
                Output = Act.Output;
            }
        }

        public override void Backward(Tensor outputgrad)
        {
            if (Act != null)
            {
                Act.Backward(outputgrad);
                outputgrad = Act.Input.Grad;
            }

            Input.Grad = Dot(outputgrad, base["w"].Data.Transpose());
            this["w"].Grad = Dot(Input.Data.Transpose(), outputgrad);
            if (UseBias)
                this["b"].Grad = Sum(outputgrad, 0);
        }
    }
}
