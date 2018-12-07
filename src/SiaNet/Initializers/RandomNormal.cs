using System;
using System.Collections.Generic;
using System.Text;
using SiaNet.Backend;

namespace SiaDNN.Initializers
{
    public class RandomNormal : BaseInitializer
    {
        public string Name
        {
            get
            {
                return "random_normal";
            }
        }

        public float Mean { get; set; }

        public float StdDev { get; set; }

        public RandomNormal(float mean = 0f, float stddev = 0.05f)
        {
            Mean = mean;
            StdDev = stddev;
        }

        public override void Operator(string name, NDArray array)
        {
            NDArray.SampleGaussian(Mean, StdDev, array);
        }

    }
}
