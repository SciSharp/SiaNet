using System;
using System.Collections.Generic;
using System.Text;

namespace SiaDNN.Initializers
{
    public class LecunNormal : VarianceScaling
    {
        public LecunNormal()
            :base(1, "fan_in", "normal")
        {
            Name = "lecun_normal";
        }
    }
}
