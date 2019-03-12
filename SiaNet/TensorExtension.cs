using SiaNet.Engine;
using System;
using System.Collections.Generic;
using System.Text;

namespace SiaNet
{
    internal static class TensorExtension
    {
        internal static Parameter ToParameter(this Tensor t, string name = "")
        {
            return Parameter.Create(t, name);
        }
    }
}
