using SiaNet.Backend;
using System;
using System.Collections.Generic;
using System.Text;

namespace SiaNet.Layers.Activations
{
    public class Selu : BaseLayer, ILayer
    {
        public float Alpha { get; set; }

        public Selu(float alpha=1)
            : base("elu")
        {
            Alpha = alpha;
        }

        public Symbol Build(Symbol x)
        {
            return new Operator("LeakyReLU").SetParam("act_type", "selu")
                                            .SetInput("data", x)
                                            .CreateSymbol(ID);
        }
    }
}
