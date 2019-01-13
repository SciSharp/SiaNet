using System;
using System.Collections.Generic;
using System.Text;
using TensorSharp;
using TensorSharp.Expression;

namespace SiaNet.Layers.Activations
{
    public class Selu : Elu
    {
        private static float alpha = 1.6732632423543772848170429916717f;
        private static float scale = 1.0507009873554804934193349852946f;

        public Selu()
            : base(alpha)
        {
            Name = "selu";
        }

        public override void Forward(Variable x)
        {
            Input = x;
           
            base.Forward(x);
            Output = scale * Output;
        }

        public override void Backward(Tensor outputgrad)
        {
            base.Backward(outputgrad);
            Input.Grad = scale * Input.Grad;
        }
    }
}
