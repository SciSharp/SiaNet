using CNTK;
using SiaNet.Regularizers;

namespace SiaNet.Optimizers
{
    public abstract class OptimizerBase
    {
        protected OptimizerBase(double learningRate, RegularizerBase regularizer)
        {
            LearningRate = learningRate;
            Regularizer = regularizer;
        }

        public double LearningRate { get; protected set; }
        public RegularizerBase Regularizer { get; protected set; }

        protected AdditionalLearningOptions GetAdditionalLearningOptions()
        {
            return Regularizer?.GetAdditionalLearningOptions() ?? new AdditionalLearningOptions();
        }

        internal abstract Learner ToLearner(Function model);
    }
}