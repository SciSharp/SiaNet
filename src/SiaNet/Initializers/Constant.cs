using System;
using System.Collections.Generic;
using System.Text;
using SiaNet.Backend;

namespace SiaDNN.Initializers
{
    public class Constant : BaseInitializer
    {
        public string Name
        {
            get
            {
                return "constant";
            }
        }

        public float Value { get; set; }

        public Constant(float value)
        {
            Value = value;
        }

        public override void Operator(string name, NDArray array)
        {
            array.Set(this.Value);
        }

    }
}
