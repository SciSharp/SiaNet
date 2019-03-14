namespace SiaNet.Initializers
{
    /// <summary>
    /// LeCun normal initializer. It draws samples from a truncated normal distribution centered on 0 with stddev = sqrt(1 / fan_in) where fan_in is the number of input units in the weight tensor.
    /// </summary>
    /// <seealso cref="SiaNet.Initializers.VarianceScaling" />
    public class LecunNormal : VarianceScaling
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="LecunNormal"/> class.
        /// </summary>
        /// <param name="seed">Used to seed the random generator.</param>
        public LecunNormal(int? seed = null)
            :base(1, "fan_in", "normal", seed)
        {
            Name = "lecun_normal";
        }
    }
}
