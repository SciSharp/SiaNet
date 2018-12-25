using System;
using System.Collections.Generic;
using System.Text;

namespace SiaNet.Initializers
{
    public class GlorotNormal : VarianceScaling
    {
        public GlorotNormal()
           : base(1, "fan_avg", "normal")
        {
            Name = "glorot_uniform";
        }
    }
}
