using System;
using System.Collections.Generic;

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
        public Dictionary<string, object> Params { get; set; } = new Dictionary<string, object>();

        protected T GetParam<T>(string name)
        {
            object o;
            if (Params.TryGetValue(name, out o) && o != null)
            {
                if (typeof(T).IsValueType)
                {
                    o = Convert.ChangeType(o, typeof(T));
                }
                return (T)o;
            }
            return default(T);
        }

        protected void SetParam<T>(string name, T value)
        {
            Params[name] = value;
        }
    }
}