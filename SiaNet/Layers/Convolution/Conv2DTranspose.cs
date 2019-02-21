using SiaNet.Constraints;
using SiaNet.Engine;
using SiaNet.Initializers;
using SiaNet.Regularizers;
using System;
using System.Collections.Generic;
using System.Text;

namespace SiaNet.Layers
{
    public class Conv2DTranspose : BaseLayer
    {
        public Conv2DTranspose(uint filters, Tuple<uint, uint> kernalSize, uint strides = 1, PaddingType padding = PaddingType.Same, Tuple<uint, uint> dialationRate = null,
                                ActType activation = ActType.Linear, BaseInitializer kernalInitializer = null, BaseRegularizer kernalRegularizer = null,
                                BaseConstraint kernalConstraint = null, bool useBias = true, BaseInitializer biasInitializer = null, BaseRegularizer biasRegularizer = null,
                                BaseConstraint biasConstraint = null)
            : base("conv2d_transpose")
        {
            Name = "conv2d_transpose";
        }


        public override void Forward(Tensor x)
        {
            throw new NotImplementedException();
        }

        public override void Backward(Tensor outputgrad)
        {
            throw new NotImplementedException();
        }
    }
}
