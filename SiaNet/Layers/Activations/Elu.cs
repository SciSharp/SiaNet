using System;
using System.Collections.Generic;
using System.Text;
using TensorSharp;
using TensorSharp.Expression;

namespace SiaNet.Layers.Activations
{
    public class Elu : BaseLayer
    {
        private float _alpha;
        public Elu(float alpha = 1)
            : base("elu")
        {
            _alpha = alpha;
        }

        public override void Forward(Variable x)
        {
            Input = x;
            var keepElements = x.Data > 0;
            
            var keepElements_Exp = x.Data <= 0;
            keepElements_Exp.Print();
            var d = _alpha * (Exp(x.Data * keepElements_Exp) - 1);
            (x.Data + keepElements - x.Data - keepElements_Exp).Print();
            Output = (x.Data + keepElements - x.Data - keepElements_Exp) + (1 - keepElements) * d;
        }

        public override void Backward(Tensor outputgrad)
        {
            var keepElements = Output.TVar() > 0;
            var d = _alpha * (Output.TVar().Exp() - 1);
            Input.Grad = (outputgrad.TVar().CMul(keepElements) + (1 - keepElements).CMul(d)).Evaluate();
        }
    }
}
