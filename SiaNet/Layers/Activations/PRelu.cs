using SiaNet.Constraints;
using SiaNet.Initializers;
using SiaNet.Regularizers;
using System;
using System.Collections.Generic;
using System.Text;
using System.Linq;
using TensorSharp;
using TensorSharp.Expression;

namespace SiaNet.Layers.Activations
{
    public class PRelu : BaseLayer
    {
        public BaseInitializer AlphaInitializer { get; set; }

        public BaseRegularizer AlphaRegularizer { get; set; }

        public BaseConstraint AlphaConstraint { get; set; }

        private Relu pos_relu;
        private Relu neg_relu;

        public PRelu(BaseInitializer alphaInitializer = null, BaseRegularizer alphaRegularizer=null, BaseConstraint alphaConstraint = null, params long[] sharedAxes)
            : base("prelu")
        {
            AlphaInitializer = alphaInitializer ?? new Zeros();
            AlphaRegularizer = alphaRegularizer;
            AlphaConstraint = alphaConstraint;
            pos_relu = new Relu();
            neg_relu = new Relu();
        }

        public override void Forward(Parameter x)
        {
            //ToDo: Implement shared axes
            Input = x;
            long[] paramShape = x.Data.Shape.ToList().Skip(1).ToArray();

            Parameter alpha = BuildParam("a", paramShape, x.Data.ElementType, AlphaInitializer, AlphaConstraint, AlphaRegularizer);
            pos_relu.Forward(x);
            var pos = pos_relu.Output;

            Parameter negX = Parameter.Create(-1 * x.Data);
            neg_relu.Forward(negX);
            var neg = -1f * alpha.Data * neg_relu.Output;
            Output = pos + neg;
        }

        public override void Backward(Tensor outputgrad)
        {
            pos_relu.Backward(outputgrad);
            neg_relu.Backward(outputgrad);

            Input.Grad = pos_relu.Input.Grad - Params["a"].Data * neg_relu.Input.Grad;
        }
    }
}
