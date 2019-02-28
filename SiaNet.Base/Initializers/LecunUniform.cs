using System;
using System.Collections.Generic;
using System.Text;

namespace SiaNet.Initializers
{
    public class LecunUniform : VarianceScaling
    {
        public LecunUniform(int? seed = null)
            :base(1, "fan_in", "uniform", seed)
        {
            Name = "lecun_uniform";
        }
    }
}
