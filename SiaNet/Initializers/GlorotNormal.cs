using System;
using System.Collections.Generic;
using System.Text;

namespace SiaNet.Initializers
{
    public class GlorotNormal : VarianceScaling
    {
        public GlorotNormal(int? seed = null)
           : base(1, "fan_avg", "normal", seed)
        {
            Name = "glorot_uniform";
        }
    }
}
