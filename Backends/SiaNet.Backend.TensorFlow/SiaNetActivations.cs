using SiaNet.Engine;
using SiaNet.Engine.Layers;
using System;
using System.Collections.Generic;
using System.Text;

namespace SiaNet.Backend.TensorFlowLib
{
    public class SiaNetActivations : ActivationFunc
    {
        SiaNetBackend backend = null;
        public SiaNetActivations(IBackend backend)
            : base(backend)
        {
            backend = K;
        }

        //public override Tensor ReluForward(Tensor x)
        //{
        //    return backend.Out(backend.tf.Relu(backend.In(x)));
        //}

        //public override Tensor ReluBackward(Tensor x, Tensor outputgrad)
        //{
        //    return backend.Out(backend.tf.ReluGrad(backend.In(x), backend.In(x)));
        //}

    }
}
