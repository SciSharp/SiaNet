using SiaNet.Engine;
using System;
using TensorFlow;

namespace SiaNet.Backend.TensorFlowLib
{
    public class TensorFlowTensor : Tensor
    {
        public TFTensor InternalTensor;

        public TensorFlowTensor()
        {
            K = new TensorFlowBackend();
        }

        public TensorFlowTensor(TFTensor arr)
        {
            InternalTensor = arr;
            K = new TensorFlowBackend();
        }

        public override string Name
        {
            get;
            set;
        }
    }
}
