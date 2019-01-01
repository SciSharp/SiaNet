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

            var p = 1 - Rate;

            noise = TVar.RandomBernoulli(new SeedSource(), p, Global.Device, x.Data.ElementType, x.Data.Shape)
                            .Div(p)
                            .Evaluate();

            Output = Output.TVar().CMul(noise).Evaluate();
        }

        public override void Backward(Tensor outputgrad)
        {
            Input.Grad = outputgrad;
            Input.Grad = Input.Grad.TVar().CMul(noise).Evaluate();
        }
    }
}
