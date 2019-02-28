using System;
using System.Collections.Generic;
using System.Text;

namespace SiaNet.Initializers
{
    public class HeUniform : VarianceScaling
    {
        public HeUniform(int? seed = null)
            :base(2, "fan_in", "uniform", seed)
        {
            Name = "he_uniform";
        }
    }
}
