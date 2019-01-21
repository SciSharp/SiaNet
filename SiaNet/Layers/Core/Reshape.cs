using System;
using System.Collections.Generic;
using System.Text;
using TensorSharp;
using TensorSharp.Expression;

namespace SiaNet.Layers
{
    public class Reshape : BaseLayer
    {
        /// <summary>
        /// Target new shape. One and only one dim can be 0, in which case it will be inferred from the rest of dims
        /// </summary>
        public long[] TargetShape { get; set; }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="targetShape"></param>
        /// <param name="reverse"></param>
        public Reshape(long[] targetShape)
            : base("reshape")
        {
            TargetShape = targetShape;
        }

        public override void Forward(Tensor x)
        {
            Input = x.ToParameter();

            Output = x.View(TargetShape);
        }

        public override void Backward(Tensor outputgrad)
        {
        }
    }
}
