using SiaNet.Engine;
using SiaNet.Engine.Layers;
using System;
using System.Collections.Generic;
using System.Text;

namespace SiaNet.Backend.TensorSharp
{
    public class TensorSharpActivations : ActivationFunc
    {
        public TensorSharpActivations(IBackend backend)
            : base(backend)
        {

        }
    }
}
