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
            K = new TensorSharpBackend();
        }

        public NDArrayTensor(NDArray arr)
        {
            InternalTensor = arr;
            K = new TensorSharpBackend();
        }

        public override string Name
        {
            get;
            set;
        }
    }
}
