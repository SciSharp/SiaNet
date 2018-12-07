using SiaNet.Backend;
using System;
using System.Collections.Generic;
using System.Text;

namespace SiaNet.Layers
{
    public interface ILayer
    {
        Symbol Build(Symbol x);
    }
}
