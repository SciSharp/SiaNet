using SiaNet.Constraints;
using SiaNet.Initializers;
using SiaNet.Regularizers;
using System;
using System.Collections.Generic;
using System.Text;
using TensorSharp;

namespace SiaNet.Layers
{
    public class Conv3DTranspose : Conv3D
    {
        public Conv3DTranspose(uint filters, Tuple<uint, uint, uint> kernalSize, uint strides = 1, PaddingType padding = PaddingType.Same, Tuple<uint, uint, uint> dialationRate = null,
                                ActivationType activation = ActivationType.Linear, BaseInitializer kernalInitializer = null,
                                 BaseRegularizer kernalRegularizer = null, BaseConstraint kernalConstraint = null,
                                bool useBias = true, BaseInitializer biasInitializer = null, BaseRegularizer biasRegularizer = null, BaseConstraint biasConstraint = null)
            : base(filters, kernalSize, strides, padding, dialationRate, activation, kernalInitializer, kernalRegularizer, kernalConstraint, useBias, biasInitializer
                  , biasRegularizer, biasConstraint)
        {
            Name = "conv3d_transpose";
        }

        public override void Forward(Parameter x)
        {
            Forward(x);
            Output = Output.Transpose();
        }

        public override void Backward(Tensor outputgrad)
        {
            outputgrad = outputgrad.Transpose();
            Backward(outputgrad);
        }
    }
}
