using System.Linq;
using CNTK;

namespace SiaNet.Model.Optimizers
{
    /// <summary>
    ///     Adagrad is an algorithm for gradient-based optimization that does just this: It adapts the learning rate to the
    ///     parameters, performing larger updates for infrequent and smaller updates for frequent parameters
    /// </summary>
    /// <seealso cref="SiaNet.Model.Optimizers.OptimizerBase" />
    public class AdaGrad : OptimizerBase
    {
        /// <summary>
        ///     Initializes a new instance of the <see cref="AdaGrad" /> class.
        /// </summary>
        /// <param name="learningRate">The learning rate.</param>
        public AdaGrad(double learningRate = 0.01, Regulizers regulizers = null) :
            base(learningRate, regulizers)
        {
        }

        /// <inheritdoc />
        internal override Learner ToLearner(Function model)
        {
            var learningRatePerSample = new TrainingParameterScheduleDouble(LearningRate, 1);

            return CNTKLib.AdaGradLearner(new ParameterVector(((CNTK.Function) model).Parameters().ToArray()),
                learningRatePerSample, false, GetAdditionalLearningOptions());
        }
    }
}