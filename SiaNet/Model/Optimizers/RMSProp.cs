using System.Linq;
using CNTK;

namespace SiaNet.Model.Optimizers
{
    /// <summary>
    ///     RMSprop is an unpublished, adaptive learning rate method proposed by Geoff Hinton. This optimizer is usually a good
    ///     choice for recurrent neural networks.
    /// </summary>
    /// <seealso cref="OptimizerBase" />
    public class RMSProp : OptimizerBase
    {
        /// <summary>
        ///     Initializes a new instance of the <see cref="RMSProp" /> class.
        /// </summary>
        /// <param name="learningRate">The learning rate.</param>
        /// <param name="gamma">The gamma.</param>
        /// <param name="inc">The inc.</param>
        /// <param name="dec">The decimal.</param>
        /// <param name="min">The minimum.</param>
        /// <param name="max">The maximum.</param>
        public RMSProp(
            double learningRate = 0.001,
            double gamma = 0.9,
            double inc = 2,
            double dec = 0.01,
            double min = 0.01,
            double max = 1,
            Regulizers regulizers = null) :
            base(learningRate, regulizers)
        {
            Gamma = gamma;
            Increment = inc;
            Decrement = dec;
            Min = min;
            Max = max;
        }

        /// <summary>
        ///     Gets or sets the decrement.
        /// </summary>
        /// <value>
        ///     The decrement.
        /// </value>
        public double Decrement { get; set; }

        /// <summary>
        ///     Gets or sets the gamma.
        /// </summary>
        /// <value>
        ///     The gamma.
        /// </value>
        public double Gamma { get; set; }

        /// <summary>
        ///     Gets or sets the increment.
        /// </summary>
        /// <value>
        ///     The increment.
        /// </value>
        public double Increment { get; set; }

        /// <summary>
        ///     Gets or sets the maximum.
        /// </summary>
        /// <value>
        ///     The maximum.
        /// </value>
        public double Max { get; set; }

        /// <summary>
        ///     Gets or sets the minimum.
        /// </summary>
        /// <value>
        ///     The minimum.
        /// </value>
        public double Min { get; set; }

        /// <inheritdoc />
        internal override Learner ToLearner(Function model)
        {
            var learningRatePerSample = new TrainingParameterScheduleDouble(LearningRate, 1);

            return CNTKLib.RMSPropLearner(new ParameterVector(((CNTK.Function) model).Parameters().ToArray()),
                learningRatePerSample, Gamma, Increment, Decrement, Max, Min, false, GetAdditionalLearningOptions());
        }
    }
}