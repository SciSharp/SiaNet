using SiaNet.Backend;
using System;
using System.Collections.Generic;
using System.Text;

namespace SiaNet.Layers.Activations
{
    public class LeakyReLU : BaseLayer, ILayer
    {
        public float Alpha { get; set; }

        public LeakyReLU(float alpha=0.3f)
            : base("leakyrelu")
        {
            Alpha = alpha;
        }

        public Symbol Build(Symbol x)
        {
            return new Operator("LeakyReLU").SetParam("act_type", "leaky")
                                            .SetInput("data", x)
                                            .SetParam("slope", Alpha)
                                            .CreateSymbol(ID);
        }
    }
}
