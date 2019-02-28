using SiaNet.Engine;
using System;
using System.Collections.Generic;
using System.Text;

namespace SiaNet.Backend.TensorSharp
{
    public class NDArrayTensor : Tensor
    {
        public NDArray InternalTensor;

        public NDArrayTensor()
        {
            K = new SiaNetBackend();
        }

        public NDArrayTensor(NDArray arr)
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
