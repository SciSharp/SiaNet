namespace SiaNet.Initializers
{
    /// <summary>
    /// Glorot uniform initializer, also called Xavier uniform initializer.
    /// <para>
    /// It draws samples from a uniform distribution within[-limit, limit] where limit is sqrt(6 / (fan_in + fan_out)) where fan_in is the number of input units in the weight tensor and fan_out is the number of output units in the weight tensor.
    /// </para>
    /// </summary>
    /// <seealso cref="SiaNet.Initializers.VarianceScaling" />
    public class GlorotUniform : VarianceScaling
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="GlorotUniform"/> class.
        /// </summary>
        /// <param name="seed">Used to seed the random generator.</param>
        public GlorotUniform(int? seed = null)
           : base(1, "fan_avg", "uniform", seed)
        {
            Name = "glorot_uniform";
        }
    }
}
