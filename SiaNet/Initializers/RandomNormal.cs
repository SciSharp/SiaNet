namespace SiaNet.Initializers
{
    using SiaNet.Engine;

    /// <summary>
    /// Initializer that generates tensors with a normal distribution.
    /// </summary>
    /// <seealso cref="SiaNet.Initializers.BaseInitializer" />
    public class RandomNormal : BaseInitializer
    {
        /// <summary>
        /// Mean of the random values to generate.
        /// </summary>
        /// <value>
        /// The mean value.
        /// </value>
        public float MeanVal { get; set; }

        /// <summary>
        /// Standard deviation of the random values to generate.
        /// </summary>
        /// <value>
        /// The standard dev.
        /// </value>
        public float StdDev { get; set; }

        /// <summary>
        /// Used to seed the random generator.
        /// </summary>
        /// <value>
        /// The seed.
        /// </value>
        public int? Seed { get; set; }

        /// <summary>
        /// Initializes a new instance of the <see cref="RandomNormal"/> class.
        /// </summary>
        /// <param name="mean">Mean of the random values to generate.</param>
        /// <param name="stddev">Standard deviation of the random values to generate.</param>
        /// <param name="seed">Used to seed the random generator.</param>
        public RandomNormal(float mean = 0f, float stddev = 0.05f, int? seed = null)
            :base ("random_normal")
        {
            MeanVal = mean;
            StdDev = stddev;
            Seed = seed;
        }

        /// <summary>
        /// Generates a tensor with specified shape.
        /// </summary>
        /// <param name="shape">The shape of the tensor.</param>
        /// <returns></returns>
        public override Tensor Generate(params long[] shape)
        {
            return K.RandomNormal(shape, MeanVal, StdDev, Seed);
        }

    }
}
