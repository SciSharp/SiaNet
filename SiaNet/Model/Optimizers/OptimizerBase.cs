using CNTK;

namespace SiaNet.Model.Optimizers
{
    public abstract class OptimizerBase
    {
        public double LearningRate { get; protected set; }
        public Regulizers Regulizer { get; protected set; }

        protected OptimizerBase(double learningRate, Regulizers regulizer)
        {
            LearningRate = learningRate;
            Regulizer = regulizer;
        }

        protected AdditionalLearningOptions GetAdditionalLearningOptions()
        {
            AdditionalLearningOptions options = new AdditionalLearningOptions();

            if (Regulizer != null)
            {
                if (Regulizer.IsL1)
                    options.l1RegularizationWeight = Regulizer.L1;
                if (Regulizer.IsL2)
                    options.l1RegularizationWeight = Regulizer.L2;

                options.gradientClippingWithTruncation = Regulizer.GradientClippingWithTruncation;
                if (Regulizer.GradientClippingThresholdPerSample.HasValue)
                    options.gradientClippingThresholdPerSample = Regulizer.GradientClippingThresholdPerSample.Value;
            }

            return options;
        }

        internal abstract Learner ToLearner(Function model);
    }
}