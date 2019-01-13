using System;
using System.Collections.Generic;
using System.Text;
using TensorSharp;
using TensorSharp.Expression;

namespace SiaNet.Layers.Activations
{
    public class HardSigmoid : BaseLayer
    {
        public HardSigmoid()
            : base("hard_sigmoid")
        {
        }

        public override void Forward(Variable x)
        {
            Input = x;
            Output = Tensor.Constant(0, Global.Device, DType.Float32, Input.Grad.Shape);
            for (int i = 0; i < Input.Data.Shape[0]; i++)
            {
                for (int j = 0; j < Input.Data.Shape[0]; j++)
                {
                    float d = Input.Data.GetElementAsFloat(i, j);
                    float r = 0;
                    if(d > 2.5)
                    {
                        r = 1;
                    }
                    else if(d >= -2.5 && d<=2.5)
                    {
                        r = (0.2f * d) + 0.5f;
                    }

                    Output.SetElementAsFloat(r, i, j);
                }
            }
        }

        public override void Backward(Tensor outputgrad)
        {
            Input.Grad = Tensor.Constant(0, Global.Device, DType.Float32, Input.Grad.Shape);
            for (int i = 0; i < Input.Data.Shape[0]; i++)
            {
                for (int j = 0; j < Input.Data.Shape[0]; j++)
                {
                    float d = Input.Data.GetElementAsFloat(i, j);
                    float r = 0;
                    if (d > 2.5)
                    {
                        r = 0;
                    }
                    else if (d >= -2.5 && d <= 2.5)
                    {
                        r = d > 0 ? -0.2f : 0.2f;
                    }

                    Input.Grad.SetElementAsFloat(r, i, j);
                }
            }
        }
    }
}
