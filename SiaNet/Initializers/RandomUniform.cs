namespace SiaNet.Initializers
{
    using SiaNet.Engine;

    /// <summary>
    /// Initializer that generates tensors with a uniform distribution.
    /// </summary>
    /// <seealso cref="SiaNet.Initializers.BaseInitializer" />
    public class RandomUniform : BaseInitializer
    {
        /// <summary>
        /// Lower bound of the range of random values to generate.
        /// </summary>
        /// <value>
        /// The minimum value.
        /// </value>
        public float MinVal { get; set; }

        /// <summary>
        /// Upper bound of the range of random values to generate.
        /// </summary>
        /// <value>
        /// The maximum value.
        /// </value>
        public float MaxVal { get; set; }

        /// <summary>
        /// Used to seed the random generator.
        /// </summary>
        /// <value>
        /// The seed.
        /// </value>
        public int? Seed { get; set; }

        /// <summary>
        /// Initializes a new instance of the <see cref="RandomUniform"/> class.
        /// </summary>
        /// <param name="minval">Lower bound of the range of random values to generate.</param>
        /// <param name="maxval">Upper bound of the range of random values to generate.</param>
        /// <param name="seed">Used to seed the random generator.</param>
        public RandomUniform(float minval = 0f, float maxval = 0.05f, int? seed = null)
            :base("random_uniform")
        {
            MinVal = minval;
            MaxVal = maxval;
            Seed = seed;
        }

        /// <summary>
        /// Generates a tensor with specified shape.
        /// </summary>
        /// <param name="shape">The shape of the tensor.</param>
        /// <returns></returns>
        public override Tensor Generate(params long[] shape)
        {
            return K.RandomUniform(shape, MinVal, MaxVal, Seed);
        }
    }
}
