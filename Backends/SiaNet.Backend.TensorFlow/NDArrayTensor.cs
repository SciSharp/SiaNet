using SiaNet.Engine;
using System;
using Tensorflow;
using Tensor = SiaNet.Engine.Tensor;

namespace SiaNet.Backend.TensorFlowLib
{
    public class NDArrayTensor : Tensor
    {
        public Tensorflow.Tensor InternalTensor;

        public NDArrayTensor()
        {
            K = new SiaNetBackend();
        }

        public NDArrayTensor(Tensorflow.Tensor arr)
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
