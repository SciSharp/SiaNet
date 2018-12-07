using System;
using System.Collections.Generic;
using System.Text;
using SiaNet.Backend;

namespace SiaDNN.Constraints
{
    public class UnitNorm : BaseConstraint
    {
        public uint Axis;

        public UnitNorm(uint axis = 0)
        {
            Axis = axis;
        }

        public override NDArray Call(NDArray w)
        {
            w = w / NDArray.Sqrt(NDArray.Sum(w, new Shape(Axis), true));
            return w;
        }
    }
}
