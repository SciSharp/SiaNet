using Newtonsoft.Json;
using SiaNet.NN;

namespace SiaNet.Model.Layers
{
    /// <summary>
    ///     Dropout consists in randomly setting a fraction rate of input units to 0 at each update during training time, which
    ///     helps prevent overfitting.
    /// </summary>
    /// <seealso cref="LayerBase" />
    public class Dropout : LayerBase
    {
        /// <summary>
        ///     Initializes a new instance of the <see cref="Dropout" /> class.
        /// </summary>
        /// <param name="rate">A float value between 0 and 1. Fraction of the input units to drop.</param>
        public Dropout(double rate)
            : this()
        {
            Rate = rate;
        }

        /// <summary>
        ///     Initializes a new instance of the <see cref="Dropout" /> class.
        /// </summary>
        internal Dropout()
        {
        }

        /// <summary>
        ///     A float value between 0 and 1. Fraction of the input units to drop.
        /// </summary>
        /// <value>
        ///     The rate.
        /// </value>
        [JsonIgnore]
        public double Rate
        {
            get => GetParam<double>("Rate");

            set => SetParam("Rate", value);
        }

        /// <inheritdoc />
        internal override Function ToFunction(Variable inputFunction)
        {
            return Basic.Dropout(inputFunction, Rate);
        }
    }
}