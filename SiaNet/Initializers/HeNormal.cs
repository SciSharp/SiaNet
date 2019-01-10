using System;
using System.Collections.Generic;
using System.Text;

namespace SiaNet.Initializers
{
    public class HeNormal : VarianceScaling
    {
        public HeNormal(int? seed = null)
            :base(2, "fan_in", "normal", seed)
        {
            Name = "he_normal";
        }
    }
}
