using System.Linq;
using CNTK;
using SiaNet.Model.Regularizers;

namespace SiaNet.Model.Optimizers
{
    /// <summary>
    ///     The Vt factor in the Adam update rule scales the gradient inversely proportionally to the ℓ2 norm of the past
    ///     gradients (via the vt−1 term) and current gradient.
    /// </summary>
    /// <seealso cref="OptimizerBase" />
    public class Adamax : OptimizerBase
    {
        /// <summary>
        ///     Initializes a new instance of the <see cref="Adamax" /> class.
        /// </summary>
        /// <param name="learningRate">The learning rate.</param>
        /// <param name="momentum">The momentum.</param>
        /// <param name="varianceMomentum">The variance momentum.</param>
        /// <param name="unitGain">if set to <c>true</c> [unit gain].</param>
        /// <param name="epsilon">The epsilon.</param>
        public Adamax(
            double learningRate = 0.002,
            double momentum = 0.9,
            double varianceMomentum = 0.999,
            bool unitGain = true,
            double epsilon = 1e-08,
            RegularizerBase regularizer = null) :
            base(learningRate, regularizer)
        {
            Momentum = momentum;
            VarianceMomentum = varianceMomentum;
            Epsilon = epsilon;
            UnitGain = unitGain;
        }

        /// <summary>
        ///     Gets or sets the epsilon.
        /// </summary>
        /// <value>
        ///     The epsilon.
        /// </value>
        public double Epsilon { get; set; }

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

        /// <summary>
        ///     Gets or sets the variance momentum.
        /// </summary>
        /// <value>
        ///     The variance momentum.
        /// </value>
        public double VarianceMomentum { get; set; }

        /// <inheritdoc />
        internal override Learner ToLearner(Function model)
        {
            var learningRatePerSample = new TrainingParameterScheduleDouble(LearningRate, 1);
            var momentumRate = new TrainingParameterScheduleDouble(Momentum, 1);
            var varianceMomentumRate = new TrainingParameterScheduleDouble(VarianceMomentum, 1);

            return CNTKLib.AdamLearner(new ParameterVector(((CNTK.Function) model).Parameters().ToArray()),
                learningRatePerSample, momentumRate, UnitGain, varianceMomentumRate, Epsilon, true,
                GetAdditionalLearningOptions());
        }
    }
}