using System;
using System.Collections.Generic;
using System.Text;
using SiaNet.Backend;

namespace SiaDNN.Initializers
{
    public class RandomUniform : BaseInitializer
    {
        public string Name
        {
            get
            {
                return "random_uniform";
            }
        }

        public float MinVal { get; set; }

        public float MaxVal { get; set; }

        public RandomUniform(float minval = 0f, float maxval = 0.05f)
        {
            MinVal = minval;
            MaxVal = maxval;
        }

        public override void Operator(string name, NDArray array)
        {
            NDArray.SampleUniform(MinVal, MaxVal, array);
        }

    }
}
