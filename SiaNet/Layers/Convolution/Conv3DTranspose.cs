using SiaNet.Constraints;
using SiaNet.Engine;
using SiaNet.Initializers;
using SiaNet.Regularizers;
using System;
using System.Collections.Generic;
using System.Text;

namespace SiaNet.Layers
{
    public class Conv3DTranspose : BaseLayer
    {
        public Conv3DTranspose(int filters, Tuple<int, int, int> kernalSize, int strides = 1, PaddingType padding = PaddingType.Same, Tuple<int, int, int> dialationRate = null,
                                ActType activation = ActType.Linear, BaseInitializer kernalInitializer = null,
                                 BaseRegularizer kernalRegularizer = null, BaseConstraint kernalConstraint = null,
                                bool useBias = true, BaseInitializer biasInitializer = null, BaseRegularizer biasRegularizer = null, BaseConstraint biasConstraint = null)
            : base("conv3d_transpose")
        {
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
