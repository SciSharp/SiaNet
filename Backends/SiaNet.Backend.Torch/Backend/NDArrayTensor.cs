using SiaNet.Engine;
using System;

namespace SiaNet.Backend.Torch
{
    public class NDArrayTensor : Tensor
    {
        public FloatTensor InternalTensor;

        public NDArrayTensor()
        {
            K = new SiaNetBackend();
        }

        public NDArrayTensor(FloatTensor arr)
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
