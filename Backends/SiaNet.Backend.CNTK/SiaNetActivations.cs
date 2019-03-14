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

        private NDArrayTensor Out(Variable x)
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

        public override Tensor ReluForward(Tensor x)
        {
            return Out(C.ReLU(In(x)));
        }

        public override Tensor ReluBackward(Tensor x, Tensor outputgrad)
        {
            var g = In(x).ToFunction().Arguments[0];
            return null;
        }
    }
}
