using System;
using System.Collections.Generic;
using System.Dynamic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SiaNet.Model.Layers
{
    /// <summary>
    /// The Activation layer
    /// </summary>
    /// <seealso cref="SiaNet.Model.LayerConfig" />
    public class Activation : LayerConfig
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="Activation"/> class.
        /// </summary>
        public Activation()
        {
            base.Name = "Activation";
            base.Params = new ExpandoObject();
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="Activation"/> class.
        /// </summary>
        /// <param name="act">The activation function to use.</param>
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
