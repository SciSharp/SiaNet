using SiaNet.Engine;
using System;
using System.Collections.Generic;
using System.Text;

namespace SiaNet.Backend.ArrayFire
{
    public class NDArrayTensor : Tensor
    {
        public NDArray InternalTensor;

        public NDArrayTensor()
        {
            K = new ArrayFireBackend();
        }

        public NDArrayTensor(NDArray arr)
        {
            InternalTensor = arr;
            K = new ArrayFireBackend();
        }

        public override string Name
        {
            get;
            set;
        }
    }
}
