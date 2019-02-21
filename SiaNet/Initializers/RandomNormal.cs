using SiaNet.Engine;
using System;
using System.Collections.Generic;
using System.Text;

namespace SiaNet.Initializers
{
    public class RandomNormal : BaseInitializer
    {
        public float MeanVal { get; set; }

        public float StdDev { get; set; }

        public int? Seed { get; set; }

        public RandomNormal(float mean = 0f, float stddev = 0.05f, int? seed = null)
            :base ("random_normal")
        {
            MeanVal = mean;
            StdDev = stddev;
            Seed = seed;
        }

        public override Tensor Operator(params long[] shape)
        {
            return K.RandomNormal(shape, MeanVal, StdDev, Seed);
        }

    }
}
