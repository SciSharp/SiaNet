namespace SiaNet.Initializers
{
    using SiaNet.Engine;
    using System;

    /// <summary>
    /// Initializer capable of adapting its scale to the shape of weights. With distribution="normal", samples are drawn from a truncated normal distribution centered on zero, with stddev = sqrt(scale / n) where n is:
    /// <list type="bullet">
    /// <item>
    /// <description>number of input units in the weight tensor, if mode = "fan_in"</description>
    /// </item>
    /// <item>
    /// <description>number of output units, if mode = "fan_out"</description>
    /// </item>
    /// <item>
    /// <description>average of the numbers of input and output units, if mode = "fan_avg"</description>
    /// </item>
    /// </list>
    /// <para>
    /// With distribution = "uniform", samples are drawn from a uniform distribution within[-limit, limit], with limit = sqrt(3 * scale / n).
    /// </para>
    /// </summary>
    /// <seealso cref="SiaNet.Initializers.BaseInitializer" />
    public class VarianceScaling : BaseInitializer
    {
        /// <summary>
        /// Scaling factor (positive float).
        /// </summary>
        /// <value>
        /// The scale.
        /// </value>
        public float Scale { get; set; }

        /// <summary>
        /// One of "fan_in", "fan_out", "fan_avg".
        /// </summary>
        /// <value>
        /// The mode.
        /// </value>
        public string Mode { get; set; }

        /// <summary>
        /// Random distribution to use. One of "normal", "uniform".
        /// </summary>
        /// <value>
        /// The distribution.
        /// </value>
        public string Distribution { get; set; }

        /// <summary>
        /// Used to seed the random generator.
        /// </summary>
        /// <value>
        /// The seed.
        /// </value>
        public int? Seed { get; set; }

        /// <summary>
        /// Initializes a new instance of the <see cref="VarianceScaling"/> class.
        /// </summary>
        /// <param name="scale">Scaling factor (positive float).</param>
        /// <param name="mode">One of "fan_in", "fan_out", "fan_avg".</param>
        /// <param name="distribution">Random distribution to use. One of "normal", "uniform".</param>
        /// <param name="seed">Used to seed the random generator.</param>
        /// <exception cref="ArgumentException">Scale must be positive value</exception>
        public VarianceScaling(float scale = 1, string mode = "fan_in", string distribution = "normal", int? seed = null)
            : base("variance_scaling")
        {
            if (scale < 1f)
            {
                throw new ArgumentException("Scale must be positive value");
            }

            ParamValidator.Validate("mode", mode, "fan_in", "fan_out", "fan_avg");
            ParamValidator.Validate("distribution", distribution, "normal", "uniform");

            Scale = scale;
            Mode = mode;
            Distribution = distribution;
            Seed = seed;
        }

        /// <summary>
        /// Generates a tensor with specified shape.
        /// </summary>
        /// <param name="shape">The shape of the tensor.</param>
        /// <returns></returns>
        public override Tensor Generate(params long[] shape)
        {
            Tensor tensor = null;
            var hwScale = 1.0f;
            if (shape.Length > 2)
            {
                for (int i = 2; i < shape.Length; ++i)
                    hwScale *= shape[i];
            }

            var @in = shape[1] * hwScale;
            var @out = shape[0] * hwScale;
            var factor = 1.0f;
            switch (Mode)
            {
                case "fan_avg":
                    factor = Scale / Math.Max(1, (@in + @out) / 2.0f);
                    break;
                case "fan_in":
                    factor = Scale / Math.Max(1, @in);
                    break;
                case "fan_out":
                    factor = Scale / Math.Max(1, @out);
                    break;
            }

            switch (Distribution)
            {
                case "uniform":
                    float limit = (float)Math.Sqrt(3f * factor);
                    tensor = K.RandomUniform(shape, -limit, limit, Seed);
                    break;
                case "normal":
                    float stddev = (float)Math.Sqrt(factor) / 0.87962566103423978f;
                    tensor = K.RandomNormal(shape, 0, stddev, Seed);
                    break;
            }

            return tensor;
        }
    }
}
