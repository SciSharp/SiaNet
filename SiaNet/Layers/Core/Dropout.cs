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
            SkipPred = true;
            Rate = rate;
        }

        public override void Forward(Tensor x)
        {
            Input = x.ToParameter();
            noise = new Tensor(x.Allocator, x.ElementType, x.Shape);
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
