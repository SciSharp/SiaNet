using SiaNet.Engine;
using SiaNet.Engine.Layers;
using System;
using System.Collections.Generic;
using System.Text;

namespace SiaNet.Backend.TensorSharp
{
    public class SiaNetActivations : ActivationFunc
    {
        public SiaNetActivations(IBackend backend)
            : base(backend)
        {

        }
    }
}
