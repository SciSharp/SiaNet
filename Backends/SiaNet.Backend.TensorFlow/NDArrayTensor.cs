using SiaNet.Engine;
using System;
using TensorFlow;

namespace SiaNet.Backend.TensorFlowLib
{
    public class NDArrayTensor : Tensor
    {
        public TFOutput InternalTensor;

        public NDArrayTensor()
        {
            K = new SiaNetBackend();
        }

        public NDArrayTensor(TFTensor arr)
        {
            InternalTensor = arr;
            K = new SiaNetBackend();
        }

        public override string Name
        {
            get;
            set;
        }
    }
}
