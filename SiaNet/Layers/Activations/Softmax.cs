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

        public override void Forward(Tensor x)
        {
            Input = x.ToParameter();
            Output = Softmax(x);
        }

        public override void Backward(Tensor outputgrad)
        {
            Input.Grad = -1 * outputgrad * Output * Input.Data;
            //Input.Grad = Tensor.Constant(0, Global.Device, DType.Float32, Input.Grad.Shape);

            //for (int i = 0; i < Input.Grad.Shape[0]; i++)
            //{
            //    for (int j = 0; j < Input.Grad.Shape[1]; j++)
            //    {
            //        float v = 0;
            //        if(i == j)
            //        {
            //            v = Output.GetElementAsFloat(i, j) * (1 - Input.Data.GetElementAsFloat(i, j));
            //        }
            //        else
            //        {
            //            v = -1 * Output.GetElementAsFloat(i, j) * Input.Data.GetElementAsFloat(i, j);
            //        }
            //    }
            //}
        }
    }
}
