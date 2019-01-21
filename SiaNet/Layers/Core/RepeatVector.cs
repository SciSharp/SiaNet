using System;
using System.Collections.Generic;
using System.Text;
using TensorSharp;
using TensorSharp.Expression;
using System.Linq;

namespace SiaNet.Layers
{
    public class RepeatVector : BaseLayer
    {
        public int NumTimes { get; set; }

        /// <summary>
        /// 
        /// </summary>
        public RepeatVector(int numTimes)
            : base("repeatvector")
        {
            NumTimes = numTimes;
        }

        public override void Forward(Tensor x)
        {
            Input = x.ToParameter();
            Output = x.RepeatTensor(NumTimes);
        }

        public override void Backward(Tensor outputgrad)
        {
            Input.Grad = outputgrad;
        }
    }
}
