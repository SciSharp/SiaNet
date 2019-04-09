using SiaNet.Engine;
using System;
using Tensor = SiaNet.Engine.Tensor;

namespace SiaNet.Backend.Torch
{
    public class NDArrayTensor : Tensor
    {
        public NDArrayTensor()
        {
            K = new SiaNetBackend();
        }

        public NDArrayTensor(object arr)
        {
            K = new SiaNetBackend();
        }

        public override string Name
        {
            get;
            set;
        }
    }
}
