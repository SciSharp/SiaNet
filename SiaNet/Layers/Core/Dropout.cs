using System;
using System.Collections.Generic;
using System.Text;
using TensorSharp;
using TensorSharp.Expression;

namespace SiaNet.Layers
{
    public class Dropout : BaseLayer
    {
        private Tensor noise;

        public float Rate { get; set; }

        public Dropout(float rate)
            :base("dropout")
        {
            Rate = rate;
        }

        public override void Forward(Variable x)
        {
            Input = x;
            noise = new Tensor(x.Data.Allocator, x.Data.ElementType, x.Data.Shape);
            var p = 1 - Rate;

            RandomBernoulli(noise, new SeedSource(), p);
            Output = noise * p;
        }

        public override void Backward(Tensor outputgrad)
        {
            Input.Grad = outputgrad;
            Input.Grad = Input.Grad * noise;
        }
    }
}
