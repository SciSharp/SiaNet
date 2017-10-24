using System;
using System.Collections.Generic;
using System.Dynamic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SiaNet.Model.Layers
{
    public class AvgPool2D : LayerConfig
    {
        public AvgPool2D()
        {
            base.Name = "AvgPool2D";
            base.Params = new ExpandoObject();
        }

        public AvgPool2D(Tuple<int, int> poolSize, Tuple<int, int> strides = null, bool padding = true)
            : this()
        {
            PoolSize = poolSize;
            Strides = strides == null ? Tuple.Create(1, 1) : strides;
            Padding = padding;
        }

        [Newtonsoft.Json.JsonIgnore]
        public Tuple<int, int> PoolSize
        {
            get
            {
                return base.Params.PoolSize;
            }

            set
            {
                base.Params.PoolSize = value;
            }
        }

        [Newtonsoft.Json.JsonIgnore]
        public Tuple<int, int> Strides
        {
            get
            {
                return base.Params.Strides;
            }

            set
            {
                base.Params.Strides = value;
            }
        }

        [Newtonsoft.Json.JsonIgnore]
        public bool Padding
        {
            get
            {
                return base.Params.Padding;
            }

            set
            {
                base.Params.Padding = value;
            }
        }
    }
}
