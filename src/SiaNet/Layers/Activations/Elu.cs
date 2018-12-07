using SiaNet.Backend;
using System;
using System.Collections.Generic;
using System.Text;

namespace SiaNet.Layers.Activations
{
    public class Elu : BaseLayer, ILayer
    {
        public float Alpha { get; set; }

        public Elu(float alpha=1)
            : base("elu")
        {
            Alpha = alpha;
        }

        public Symbol Build(Symbol x)
        {
            return new Operator("LeakyReLU").SetParam("act_type", "elu")
                                            .SetInput("data", x)
                                            .SetParam("slope", Alpha)
                                            .CreateSymbol(ID);
        }
    }
}
