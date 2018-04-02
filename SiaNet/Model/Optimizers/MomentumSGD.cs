using System.Linq;
using CNTK;

namespace SiaNet.Model.Optimizers
{
    /// <summary>
    ///     Momentum of Stochastic gradient descent optimizer.
    /// </summary>
    /// <seealso cref="OptimizerBase" />
    public class MomentumSGD : OptimizerBase
    {
        /// <summary>
        ///     Initializes a new instance of the <see cref="MomentumSGD" /> class.
        /// </summary>
        /// <param name="learningRate">The learning rate.</param>
        /// <param name="momentum">The momentum.</param>
        /// <param name="unitGain">if set to <c>true</c> [unit gain].</param>
        public MomentumSGD(
            double learningRate = 0.01,
            double momentum = 0.1,
            bool unitGain = true,
            Regulizers regulizers = null) :
            base(learningRate, regulizers)
        {
            Momentum = momentum;
            UnitGain = unitGain;
        }

        /// <summary>
        ///     Gets or sets the momentum.
        /// </summary>
        /// <value>
        ///     The momentum.
        /// </value>
        public double Momentum { get; set; }

        /// <summary>
        ///     Gets or sets a value indicating whether [unit gain].
        /// </summary>
        /// <value>
        ///     <c>true</c> if [unit gain]; otherwise, <c>false</c>.
        /// </value>
        public bool UnitGain { get; set; }

        /// <inheritdoc />
        internal override Learner ToLearner(Function model)
        {
            var learningRatePerSample = new TrainingParameterScheduleDouble(LearningRate, 1);
            var momentumPerSample = new TrainingParameterScheduleDouble(Momentum, 1);

            return CNTKLib.MomentumSGDLearner(new ParameterVector(((CNTK.Function) model).Parameters().ToArray()),
                learningRatePerSample, momentumPerSample, UnitGain, GetAdditionalLearningOptions());
        }
    }
}