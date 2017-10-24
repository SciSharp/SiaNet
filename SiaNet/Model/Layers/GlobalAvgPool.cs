using System;
using System.Collections.Generic;
using System.Dynamic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SiaNet.Model.Layers
{
    public class GlobalAvgPool1D : LayerConfig
    {
        public GlobalAvgPool1D()
        {
            base.Name = "GlobalAvgPool1D";
            base.Params = new ExpandoObject();
        }
    }
    public class GlobalAvgPool2D : LayerConfig
    {
        public GlobalAvgPool2D()
        {
            base.Name = "GlobalAvgPool2D";
            base.Params = new ExpandoObject();
        }
    }

    public class GlobalAvgPool3D : LayerConfig
    {
        public GlobalAvgPool3D()
        {
            base.Name = "GlobalAvgPool3D";
            base.Params = new ExpandoObject();
        }
    }

}
