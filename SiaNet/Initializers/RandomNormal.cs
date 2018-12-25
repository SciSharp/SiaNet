using System;
using System.Collections.Generic;
using System.Text;
using TensorSharp;
using TensorSharp.Expression;

namespace SiaNet.Initializers
{
    public class RandomNormal : BaseInitializer
    {
        public float Mean { get; set; }

        public float StdDev { get; set; }

        public RandomNormal(float mean = 0f, float stddev = 0.05f)
            :base ("random_normal")
        {
            Mean = mean;
            StdDev = stddev;
        }

        public override Tensor Operator(Tensor array)
        {
            Ops.RandomNormal(array, new SeedSource(), Mean, StdDev);
            return array;
        }

    }
}
