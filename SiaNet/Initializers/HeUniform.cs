namespace SiaNet.Initializers
{
    /// <summary>
    /// He uniform variance scaling initializer.
    ///<para>
    ///It draws samples from a uniform distribution within[-limit, limit] where limit is sqrt(6 / fan_in) where fan_in is the number of input units in the weight tensor.
    ///</para>
    /// </summary>
    /// <seealso cref="SiaNet.Initializers.VarianceScaling" />
    public class HeUniform : VarianceScaling
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="HeUniform"/> class.
        /// </summary>
        /// <param name="seed">Used to seed the random generator.</param>
        public HeUniform(int? seed = null)
            :base(2, "fan_in", "uniform", seed)
        {
            Name = "he_uniform";
        }
    }
}
