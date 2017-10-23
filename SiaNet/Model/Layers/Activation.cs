using System;
using System.Collections.Generic;
using System.Dynamic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SiaNet.Model.Layers
{
    public class Activation : LayerConfig
    {
        public Activation()
        {
            base.Name = "Activation";
            base.Params = new ExpandoObject();
        }

        public Activation(string act)
            : this()
        {
            Act = act;
        }

        [Newtonsoft.Json.JsonIgnore]
        public string Act
        {
            get
            {
                return base.Params.Act;
            }

            set
            {
                base.Params.Act = value;
            }
        }
    }
}
