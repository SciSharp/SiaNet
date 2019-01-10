using System;
using System.Collections.Generic;
using System.Text;

namespace SiaNet.Initializers
{
    public class GlorotUniform : VarianceScaling
    {
        public GlorotUniform(int? seed = null)
           : base(1, "fan_avg", "uniform", seed)
        {
            Name = "glorot_uniform";
        }
    }
}
