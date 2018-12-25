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
            Variable weight = null;
            Variable bias = null;
            if (!Params.ContainsKey("w"))
            {
                weight = new Variable("w", x.Data.ElementType, x.Data.Sizes[1], Dim);
                weight.Data = KernalInitializer.Operator(weight.Data);
                weight.SetConstraint(KernalConstraint);
                weight.SetRegularizer(KernalRegularizer);

                Params.Add("w", weight);
            }
            else
            {
                weight = Params["w"];
            }

            if (UseBias)
            {
                if (!Params.ContainsKey("b"))
                {
                    bias = new Variable("b", x.Data.ElementType, 1, Dim);
                    bias.Data = BiasInitializer.Operator(bias.Data);
                    bias.SetConstraint(BiasConstraint);
                    bias.SetRegularizer(BiasRegularizer);

                    Params.Add("b", bias);
                }
                else
                {
                    bias = Params["b"];
                }

                Output = Variable.Create(bias.Data.TVar().Expand(x.Data.Sizes[0], Dim)
                    .Addmm(1, 1, x.Data, weight.Data)
                    .Evaluate());
            }
            else
            {
                Output = Variable.Create(x.Data.TVar().Dot(weight.Data.TVar()).Evaluate());
            }

            Activation.Forward(Output);
        }

        public override void Backward(Tensor outputgrad)
        {
            Activation.Backward(outputgrad);

            Input.Grad = outputgrad.TVar().Dot(Params["w"].Data.TVar().Transpose()).Evaluate();

            Params["w"].Grad = outputgrad.TVar().Dot(Params["w"].Data.TVar().Transpose()).Evaluate();
            if(UseBias)
                Params["b"].Grad = (Params["b"].Grad + outputgrad.TVar().Sum(0)).Evaluate();
        }
    }
}
