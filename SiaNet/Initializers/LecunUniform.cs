namespace SiaNet.Initializers
{
    /// <summary>
    /// LeCun uniform initializer.
    /// It draws samples from a uniform distribution within[-limit, limit] where limit is sqrt(3 / fan_in) where fan_in is the number of input units in the weight tensor.
    /// </summary>
    /// <seealso cref="SiaNet.Initializers.VarianceScaling" />
    public class LecunUniform : VarianceScaling
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="LecunUniform"/> class.
        /// </summary>
        /// <param name="seed">Used to seed the random generator.</param>
        public LecunUniform(int? seed = null)
            :base(1, "fan_in", "uniform", seed)
        {
            Name = "lecun_uniform";
        }
    }
}
