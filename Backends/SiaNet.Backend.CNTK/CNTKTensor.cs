using SiaNet.Engine;
using System;
using System.Collections.Generic;
using System.Text;

namespace SiaNet.Backend.CNTKLib
{
    public class CNTKTensor : Tensor
    {
        public CNTK.Function InternalTensor;

        public CNTKTensor()
        {
            K = new CNTKBackend();
        }

        public CNTKTensor(CNTK.Function arr)
        {
            InternalTensor = arr;
            K = new CNTKBackend();
        }

        public override string Name
        {
            get;
            set;
        }
    }
}
