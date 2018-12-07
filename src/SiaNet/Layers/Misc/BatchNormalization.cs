using System;
using System.Collections.Generic;
using System.Text;
using SiaDNN.Constraints;
using SiaDNN.Initializers;
using SiaNet.Backend;
using SiaNet.Regularizers;

namespace SiaNet.Layers.Misc
{
    public class BatchNormalization : BaseLayer, ILayer
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

        public BatchNormalization(int axis = -1, float momentum = 0.99f, float epsilon = 0.001f, bool center = true, bool scale=true,
                                   BaseInitializer betaInitializer = null, BaseRegularizer betaRegularizer = null, BaseConstraint betaConstraint = null, BaseInitializer gammaInitializer = null,
                                   BaseRegularizer gammaRegularizer = null, BaseConstraint gammaConstraint = null, BaseInitializer movingMeanInitializer = null, BaseInitializer movingVarianceInitializer = null)
            :base("batchnormalization")
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

        public Symbol Build(Symbol x)
        {
            var beta = UUID.GetID(ID + "_beta");
            var gamma = UUID.GetID(ID + "_beta");
            var movingMean = UUID.GetID(ID + "_mean");
            var movingVar = UUID.GetID(ID + "_var");

            InitParams.Add(beta, BetaInitializer);
            InitParams.Add(gamma, GammaInitializer);
            InitParams.Add(movingMean, MovingMeanInitializer);
            InitParams.Add(movingVar, MovingVarianceInitializer);

            ConstraintParams.Add(beta, BetaConstraint);
            ConstraintParams.Add(gamma, GammaConstraint);

            RegularizerParams.Add(beta, BetaRegularizer);
            RegularizerParams.Add(gamma, GammaRegularizer);

            return Operators.BatchNorm(ID, x, Symbol.Variable(gamma), Symbol.Variable(beta), Symbol.Variable(movingMean), Symbol.Variable(movingVar),
                                        Epsilon, Momentum, Center, Scale, false, Axis, !GlobalParam.UseCudnn);
        }
    }
}
