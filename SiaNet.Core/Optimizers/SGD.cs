using System.Linq;
using CNTK;
using SiaNet.Regularizers;

namespace SiaNet.Optimizers
{
    /// <summary>
    ///     SGD is an optimisation technique. It is an alternative to Standard Gradient Descent and other approaches like batch
    ///     training or BFGS. It still leads to fast convergence, with some advantages:
    ///     - Doesn't require storing all training data in memory (good for large training sets)
    ///     - Allows adding new data in an "online" setting
    /// </summary>
    /// <seealso cref="OptimizerBase" />
    public class SGD : OptimizerBase
    {
        /// <summary>
        ///     Initializes a new instance of the <see cref="SGD" /> class.
        /// </summary>
        /// <param name="learningRate">The learning rate.</param>
        public SGD(double learningRate = 0.01, RegularizerBase regularizer = null) :
            base(learningRate, regularizer)
        {
        }

        /// <inheritdoc />
        internal override Learner ToLearner(Function model)
        {
            var learningRatePerSample = new TrainingParameterScheduleDouble(LearningRate, 1);

            return CNTKLib.SGDLearner(new ParameterVector(((CNTK.Function) model).Parameters().ToArray()),
                learningRatePerSample, GetAdditionalLearningOptions());
        }
    }
}