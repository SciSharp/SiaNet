using SiaNet.Engine;
using SiaNet.Engine.Layers;
using System;
using System.Collections.Generic;
using System.Text;

namespace SiaNet.Backend.ArrayFire
{
    public class ArrayFireActivations : ActivationFunc
    {
        public ArrayFireActivations(IBackend backend)
            : base(backend)
        {

        }
    }
}
