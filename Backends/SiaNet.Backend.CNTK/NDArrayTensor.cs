using SiaNet.Engine;
using System;
using System.Collections.Generic;
using System.Text;

namespace SiaNet.Backend.CNTKLib
{
    public class NDArrayTensor : Tensor
    {
        public CNTK.Variable InternalTensor;

        public NDArrayTensor()
        {
            K = new SiaNetBackend();
        }

        public NDArrayTensor(CNTK.Variable arr)
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
