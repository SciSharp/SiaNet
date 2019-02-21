using SiaNet.Engine;
using System;
using System.Collections.Generic;
using System.Text;

namespace SiaNet
{
    public static class TensorExtension
    {
        public static Parameter ToParameter(this Tensor t, string name = "")
        {
            return Parameter.Create(t, name);
        }
    }
}
