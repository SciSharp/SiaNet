using System;
using System.Collections.Generic;
using System.Dynamic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SiaNet.Model.Layers
{
    public class GlobalMaxPool1D : LayerConfig
    {
        public GlobalMaxPool1D()
        {
            base.Name = "GlobalMaxPool1D";
            base.Params = new ExpandoObject();
        }
    }
    public class GlobalMaxPool2D : LayerConfig
    {
        public GlobalMaxPool2D()
        {
            base.Name = "GlobalMaxPool2D";
            base.Params = new ExpandoObject();
        }
    }

    public class GlobalMaxPool3D : LayerConfig
    {
        public GlobalMaxPool3D()
        {
            base.Name = "GlobalMaxPool3D";
            base.Params = new ExpandoObject();
        }
    }

}
