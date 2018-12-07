using System;
using System.Collections.Generic;
using System.Text;

namespace SiaDNN.Initializers
{
    public class GlorotUniform : VarianceScaling
    {
        public GlorotUniform()
            :base(1, "fan_avg", "uniform")
        {
            Name = "glorot_uniform";
        }
    }
}
