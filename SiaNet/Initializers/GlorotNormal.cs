namespace SiaNet.Initializers
{
    /// <summary>
    /// Glorot normal initializer, also called Xavier normal initializer.
    ///<para>
    ///It draws samples from a truncated normal distribution centered on 0 with stddev = sqrt(2 / (fan_in + fan_out)) where fan_in is the number of input units in the weight tensor and fan_out is the number of output units in the weight tensor.
    ///</para>
    /// </summary>
    /// <seealso cref="SiaNet.Initializers.VarianceScaling" />
    public class GlorotNormal : VarianceScaling
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="GlorotNormal"/> class.
        /// </summary>
        /// <param name="seed">Used to seed the random generator.</param>
        public GlorotNormal(int? seed = null)
           : base(1, "fan_avg", "normal", seed)
        {
            Name = "glorot_uniform";
        }
    }
}
