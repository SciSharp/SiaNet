using System;
using System.Collections.Generic;
using System.Text;
using TensorSharp;
using System.Linq;

namespace SiaNet.Data
{
    public class DataFrame2D : DataFrame
    {
        private long features;

        public DataFrame2D(long num_features)
            : base()
        {
            features = num_features;
        }

        public void Load(params float[] array)
        {
            underlayingVariable = Tensor.FromArray(Global.Device, array.ToArray());
            underlayingVariable.AsType(DType.Float32);
            underlayingVariable = underlayingVariable.Reshape(-1, features);
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
