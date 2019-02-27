using System;
using System.Collections.Generic;
using System.Text;
using SiaNet.Constraints;
using SiaNet.Engine;
using SiaNet.Initializers;
using SiaNet.Layers.Activations;
using SiaNet.Regularizers;

namespace SiaNet.Layers
{
    public class BatchNormalization : BaseLayer
    {
        public int Axis { get; set; }

        public float Momentum { get; set; }

        public float Epsilon { get; set; }

        public bool Center { get; set; }

        public bool Scale { get; set; }

        public BaseInitializer BetaInitializer { get; set; }

        public BaseInitializer GammaInitializer { get; set; }

        public BaseInitializer MovingMeanInitializer { get; set; }

        public BaseInitializer MovingVarianceInitializer { get; set; }

        public BaseConstraint BetaConstraint { get; set; }

        public BaseConstraint GammaConstraint { get; set; }

        public BaseRegularizer BetaRegularizer { get; set; }

        public BaseRegularizer GammaRegularizer { get; set; }

        private Parameter mu;

        private Parameter mv;

        private Tensor norm;

        public BatchNormalization(int axis = -1, float momentum = 0.99f, float epsilon = 0.001f, bool center = true, bool scale = true,
                                   BaseInitializer betaInitializer = null, BaseRegularizer betaRegularizer = null, BaseConstraint betaConstraint = null, BaseInitializer gammaInitializer = null,
                                   BaseRegularizer gammaRegularizer = null, BaseConstraint gammaConstraint = null, BaseInitializer movingMeanInitializer = null, BaseInitializer movingVarianceInitializer = null)
            : base("batchnormalization")
        {
            Axis = axis;
            Momentum = momentum;
            Epsilon = epsilon;
            Center = center;
            Scale = scale;
            BetaInitializer = betaInitializer ?? new Zeros();
            GammaInitializer = gammaInitializer ?? new Ones();
            MovingMeanInitializer = movingMeanInitializer ?? new Zeros();
            MovingVarianceInitializer = movingVarianceInitializer ?? new Ones();
            BetaConstraint = betaConstraint;
            GammaConstraint = gammaConstraint;
            BetaRegularizer = betaRegularizer;
            GammaRegularizer = gammaRegularizer;
        }

        public override void Forward(Tensor x)
        {
            base.Forward(x);
            
            Parameter beta = BuildParam("beta", x.Shape, x.ElementType, BetaInitializer, BetaConstraint, BetaRegularizer);
            Parameter gamma = BuildParam("gamma", x.Shape, x.ElementType, GammaInitializer, GammaConstraint, GammaRegularizer);

            mu = BuildParam("mm", x.Shape, x.ElementType, MovingMeanInitializer, null, null, false);
            mv = BuildParam("mv", x.Shape, x.ElementType, MovingVarianceInitializer, null, null, false);

            norm = (x - mu.Data) / K.Sqrt((mv.Data + K.Epsilon()));
            
            var @out = gamma.Data * norm + beta.Data;
            Output = K.Reshape(@out, x.Shape);
        }

        public override void Backward(Tensor outputgrad)
        {
            Tensor mm = Params["mm"].Data;
            Tensor mv = Params["mv"].Data;

            var X_mu = Input.Data - mm;
            var var_inv = 1 / K.Sqrt(mv + K.Epsilon());

            var dbeta = K.Sum(outputgrad, 0);
            var dgamma = K.Sum(outputgrad * norm, 0);

            var dnorm = outputgrad * Params["gamma"].Data;
            var dvar = K.Sum(dnorm * mu.Data, 0) * K.Pow(-0.5f * (mv + K.Epsilon()), -3 / 2);
            var dmu = K.Sum(-1 * dnorm * var_inv, 0) + K.Sum(dvar * (-2 * X_mu), 0) / Input.Data.Shape[0];

            var dX = dnorm * var_inv + (dmu / Input.Data.Shape[0]) + (dvar * (2 / Input.Data.Shape[0] * X_mu));

            Input.Grad = dX;

            Params["beta"].Grad = dgamma;
            Params["gamma"].Grad = dgamma;
        }
    }
}
