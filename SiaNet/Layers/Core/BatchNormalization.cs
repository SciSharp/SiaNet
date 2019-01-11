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

        private Variable mu;

        private Variable mv;

        private TVar norm;

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

        public override void Forward(Variable x)
        {
            Input = x;
            
            Variable beta = BuildVar("beta", x.Data.Shape, x.Data.ElementType, BetaInitializer, BetaConstraint, BetaRegularizer);
            Variable gamma = BuildVar("gamma", x.Data.Shape, x.Data.ElementType, GammaInitializer, GammaConstraint, GammaRegularizer);

            mu = BuildVar("mm", x.Data.Shape, x.Data.ElementType, MovingMeanInitializer, null, null, false);
            mv = BuildVar("mv", x.Data.Shape, x.Data.ElementType, MovingVarianceInitializer, null, null, false);

            norm = (x.Data - mu.Data.TVar()).CDiv((mv.Data.TVar() + EPSILON).Sqrt());
            
            var @out = gamma.Data.TVar().CMul(norm) + beta.Data;
            Output = @out.View(x.Data.Shape).Evaluate();
        }

        public override void Backward(Tensor outputgrad)
        {
            Tensor mm = Params["mm"].Data;
            Tensor mv = Params["mv"].Data;

            var X_mu = Input.Data.TVar() - mm.TVar();
            var var_inv = 1 / (mv.TVar() + EPSILON).Sqrt();

            var dbeta = outputgrad.TVar().Sum(0);
            var dgamma = outputgrad.TVar().CMul(norm).Sum(0);

            var dnorm = outputgrad.TVar().CMul(Params["gamma"].Data);
            var dvar = dnorm.CMul(mu.Data).Sum(0).CMul(-0.5f * (mv.TVar() + EPSILON).Pow(-3 / 2));
            var dmu = dnorm.CMul(-var_inv).Sum(0) + dvar.CMul((-2 * X_mu).Sum(0) / Input.Data.Shape[0]);

            var dX = dnorm.CMul(var_inv) + (dmu / Input.Data.Shape[0]) + (dvar.CMul(2 / Input.Data.Shape[0] * X_mu));

            Input.Grad = dX.Evaluate();
            
            Params["beta"].Grad = dgamma.Evaluate();
            Params["gamma"].Grad = dgamma.Evaluate();
        }
    }
}
