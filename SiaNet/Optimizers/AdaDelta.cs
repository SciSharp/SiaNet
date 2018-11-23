using System.Linq;
using CNTK;
using SiaNet.Regularizers;

namespace SiaNet.Optimizers
{
    /// <summary>
    ///     Adadelta is an extension of Adagrad that seeks to reduce its aggressive, monotonically decreasing learning rate.
    ///     Instead of accumulating all past squared gradients, Adadelta restricts the window of accumulated past gradients to
    ///     some fixed size w.
    /// </summary>
    /// <seealso cref="SiaNet.Model.Optimizers.OptimizerBase" />
    public class AdaDelta : OptimizerBase
    {
        /// <summary>
        ///     Initializes a new instance of the <see cref="AdaDelta" /> class.
        /// </summary>
        /// <param name="learningRate">The learning rate.</param>
        /// <param name="rho">The rho.</param>
        /// <param name="epsilon">The epsilon.</param>
        /// <param name="regularizer">The regularizer.</param>
        public AdaDelta(double learningRate = 1.0, double rho = 0.95, double epsilon = 1e-08, RegularizerBase regularizer = null) :
            base(learningRate, regularizer)
        {
            Rho = rho;
            Epsilon = epsilon;
        }

        /// <summary>
        ///     Gets or sets the epsilon.
        /// </summary>
        /// <value>
        ///     The epsilon.
        /// </value>
        public double Epsilon { get; set; }

        /// <summary>
        ///     Gets or sets the rho.
        /// </summary>
        /// <value>
        ///     The rho.
        /// </value>
        public double Rho { get; set; }

        /// <inheritdoc />
        internal override Learner ToLearner(Function modelOutput)
        {
            var learningRatePerSample = new TrainingParameterScheduleDouble(LearningRate, 1);

            return CNTKLib.AdaDeltaLearner(new ParameterVector(((CNTK.Function) modelOutput).Parameters().ToArray()),
                learningRatePerSample, Rho, Epsilon, GetAdditionalLearningOptions());
        }
    }
}