using System;
using System.Collections.Generic;
using System.Text;
using TensorSharp;
using TensorSharp.Expression;

namespace SiaNet.Layers.Activations
{
    public class Softmax : BaseLayer
    {
        public Softmax()
            : base("softmax")
        {
        }

        public override void Forward(Variable x)
        {
            Input = x;
            Output = x.Data.TVar().Softmax().Evaluate();
        }

        public override void Backward(Tensor outputgrad)
        {
            Input.Grad = Output;

            for (long i = 0; i < Input.Grad.Shape[0]; i++)
            {
                for (long j = 0; j < Input.Grad.Shape[1]; j++)
                {
                    float val = Input.Grad.GetElementAsFloat(i, j);
                    if(i == j)
                    {
                        
                    }
                }
            }
        }
    }
}
