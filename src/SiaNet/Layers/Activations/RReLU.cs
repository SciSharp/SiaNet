using SiaNet.Backend;
using System;
using System.Collections.Generic;
using System.Text;

namespace SiaNet.Layers.Activations
{
    public class RReLU : BaseLayer, ILayer
    {
        public float LowerBound { get; set; }

        public float UpperBound { get; set; }

        public RReLU(float lower_bound = 0.125f, float upper_bound=0.334f)
            : base("rrelu")
        {
            LowerBound = lower_bound;
            UpperBound = upper_bound;
        }

        public Symbol Build(Symbol x)
        {
            return new Operator("LeakyReLU").SetParam("act_type", "rrelu")
                                            .SetInput("data", x)
                                            .SetParam("lower_bound", LowerBound)
                                            .SetParam("upper_bound", UpperBound)
                                            .CreateSymbol(ID);
        }
    }
}
