using SiaNet.Engine;
using SiaNet.Engine.Layers;
using System;
using System.Collections.Generic;
using System.Text;

namespace SiaNet.Backend.Torch
{
    public class SiaNetActivations : ActivationFunc
    {
        SiaNetBackend backend = null;
        public SiaNetActivations(IBackend backend)
            : base(backend)
        {
            backend = K;
        }

    }
}
