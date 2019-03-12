namespace SiaNet.Initializers
{
    /// <summary>
    /// He normal initializer.
    /// <para>
    /// It draws samples from a truncated normal distribution centered on 0 with stddev = sqrt(2 / fan_in) where fan_in is the number of input units in the weight tensor.
    /// </para>
    /// </summary>
    /// <seealso cref="SiaNet.Initializers.VarianceScaling" />
    public class HeNormal : VarianceScaling
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="HeNormal"/> class.
        /// </summary>
        /// <param name="seed">Used to seed the random generator.</param>
        public HeNormal(int? seed = null)
            :base(2, "fan_in", "normal", seed)
        {
            Name = "he_normal";
        }
    }
}
