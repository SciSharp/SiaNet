using System;
using System.Collections.Generic;
using System.Text;
using SiaNet.Backend;

namespace SiaNet.Layers
{
    public class UpSampling : BaseLayer, ILayer
    {
        public int Scale { get; set; }

        public UpSampling(int scale = 2)
            :base("upsampling")
        {
            Scale = scale;
        }

        public Symbol Build(Symbol x)
        {
            return new Operator("UpSampling").SetInput("data", x).SetParam("scale", Scale).CreateSymbol(ID);
        }
    }
}
