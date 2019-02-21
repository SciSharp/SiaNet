using SiaNet.Engine;
using System;
using System.Collections.Generic;
using System.Text;

namespace SiaNet.Initializers
{
    public class VarianceScaling : BaseInitializer
    {
        public float Scale { get; set; }

        public string Mode { get; set; }

        public string Distribution { get; set; }

        public int? Seed { get; set; }

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

        public override Tensor Operator(params long[] shape)
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
