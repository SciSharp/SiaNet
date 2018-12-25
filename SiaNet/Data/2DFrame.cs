using System;
using System.Collections.Generic;
using System.Text;
using TensorSharp;

namespace SiaNet.Data
{
    public class DataFrame2D : DataFrame
    {
        private long cols;

        public DataFrame2D(long columnLength)
            : base(0, columnLength)
        {
            cols = columnLength;
        }

        public void Load(float[] array)
        {
            Shape = new long[] { array.LongLength / cols, cols };
            underlayingVariable = new Tensor(Global.Device, DType.Float32, Shape);
            underlayingVariable.CopyFrom(array);
        }

        public override void ToFrame(Tensor t)
        {
            if(t.DimensionCount != 2)
            {
                throw new ArgumentException("2D tensor expected");
            }

            base.ToFrame(t);
        }
    }
}
