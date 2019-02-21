using System;
using System.Collections.Generic;
using System.Text;
using System.Linq;
using SiaNet.Engine;

namespace SiaNet.Layers
{
    public class Repeat : BaseLayer
    {
        public int Axis { get; set; }

        public int NumTimes { get; set; }

        /// <summary>
        /// 
        /// </summary>
        public Repeat(int numTimes, int axis = 0)
            : base("repeatvector")
        {
            NumTimes = numTimes;
            Axis = axis;
        }

        public override void Forward(Tensor x)
        {
            Input = x.ToParameter();
            Output = K.Tile(x, NumTimes, Axis);
        }

        public override void Backward(Tensor outputgrad)
        {
            Input.Grad = outputgrad.Reshape(Input.Data.Shape);
        }
    }
}
