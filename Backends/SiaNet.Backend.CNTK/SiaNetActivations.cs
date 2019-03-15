using CNTK;
using SiaNet.Engine;
using SiaNet.Engine.Layers;
using System;
using System.Collections.Generic;
using System.Text;
using C = CNTK.CNTKLib;

namespace SiaNet.Backend.CNTKLib
{
    public class SiaNetActivations : ActivationFunc
    {
        Function act;

        private Variable In(Tensor x)
        {
            return ((NDArrayTensor)x).InternalTensor;
        }

        private NDArrayTensor Out(Parameter x)
        {
            NDArrayTensor tensor = new NDArrayTensor
            {
                InternalTensor = x
            };

            return tensor;
        }

        public SiaNetActivations(IBackend backend)
            : base(backend)
        {

        }
    }
}
