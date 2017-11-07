namespace SiaNet.Model.Optimizers
{
    using SiaNet.Common;

    /// <summary>
    /// RMSprop is an unpublished, adaptive learning rate method proposed by Geoff Hinton. This optimizer is usually a good choice for recurrent neural networks.
    /// </summary>
    /// <seealso cref="SiaNet.Model.Optimizers.BaseOptimizer" />
    public class RMSProp : BaseOptimizer
    {
        /// <summary>
        /// Gets or sets the gamma.
        /// </summary>
        /// <value>
        /// The gamma.
        /// </value>
        public double Gamma
        {
            get
            {
                return (double)AdditionalParams["Gamma"];
            }
            set
            {
                AdditionalParams["Gamma"] = value;
            }
        }

        /// <summary>
        /// Gets or sets the increment.
        /// </summary>
        /// <value>
        /// The increment.
        /// </value>
        public double Increment
        {
            get
            {
                return (double)AdditionalParams["Increment"];
            }
            set
            {
                AdditionalParams["Increment"] = value;
            }
        }

        /// <summary>
        /// Gets or sets the decrement.
        /// </summary>
        /// <value>
        /// The decrement.
        /// </value>
        public double Decrement
        {
            get
            {
                return (double)AdditionalParams["Decrement"];
            }
            set
            {
                AdditionalParams["Decrement"] = value;
            }
        }

        /// <summary>
        /// Gets or sets the minimum.
        /// </summary>
        /// <value>
        /// The minimum.
        /// </value>
        public double Min
        {
            get
            {
                return (double)AdditionalParams["Min"];
            }
            set
            {
                AdditionalParams["Min"] = value;
            }
        }

        /// <summary>
        /// Gets or sets the maximum.
        /// </summary>
        /// <value>
        /// The maximum.
        /// </value>
        public double Max
        {
            get
            {
                return (double)AdditionalParams["Max"];
            }
            set
            {
                AdditionalParams["Max"] = value;
            }
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="RMSProp"/> class.
        /// </summary>
        /// <param name="learningRate">The learning rate.</param>
        /// <param name="gamma">The gamma.</param>
        /// <param name="inc">The inc.</param>
        /// <param name="dec">The decimal.</param>
        /// <param name="min">The minimum.</param>
        /// <param name="max">The maximum.</param>
        public RMSProp(double learningRate = 0.001, double gamma = 0.9, double inc = 2, double dec = 0.01, double min = 0.01, double max = 1) :
            base(OptOptimizers.RMSProp, learningRate)
        {
            Gamma = gamma;
            Increment = inc;
            Decrement = dec;
            Min = min;
            Max = max;
        }
    }
}
