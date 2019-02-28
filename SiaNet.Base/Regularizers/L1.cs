using System;
using System.Collections.Generic;
using System.Text;

namespace SiaNet.Regularizers
{
    public class L1 : L1L2
    {
        public L1(float l = 0.01f)
            : base(l, 0)
        {
            Name = "L1";
        }
    }
}
