using System;
using System.Collections.Generic;
using System.Text;

namespace SiaNet.Initializers
{
    public class LecunNormal : VarianceScaling
    {
        public LecunNormal(int? seed = null)
            :base(1, "fan_in", "normal", seed)
        {
            Name = "lecun_normal";
        }
    }
}
