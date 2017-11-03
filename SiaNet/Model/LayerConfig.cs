using System;
using System.Collections.Generic;
using System.Dynamic;
using System.Linq;
using System.Linq.Expressions;
using System.Text;
using System.Threading.Tasks;

namespace SiaNet.Model
{
    /// <summary>
    /// Base class for the network layer
    /// </summary>
    public class LayerConfig
    {
        /// <summary>
        /// Gets or sets the name of the network layer
        /// </summary>
        /// <value>
        /// The name.
        /// </value>
        public string Name { get; set; }

        /// <summary>
        /// Gets or sets the parameters.
        /// </summary>
        /// <value>
        /// The parameters of the layer.
        /// </value>
        public dynamic Params { get; set; }
    }
}