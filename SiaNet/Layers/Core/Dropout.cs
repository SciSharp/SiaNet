using SiaNet.Engine;
using System;
using System.Collections.Generic;
using System.Text;

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
            base.Forward(x);
            
            var p = 1 - Rate;

            if (noise == null)
            {
                noise = K.RandomBernoulli(x.Shape, p);
            }

            Output = noise * p;
        }

        public override void Backward(Tensor outputgrad)
        {
            Input.Grad = outputgrad * noise;
        }
    }
}
