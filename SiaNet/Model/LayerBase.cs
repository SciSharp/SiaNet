using System;
using System.Collections.Generic;
using CNTK;

namespace SiaNet.Model
{
    /// <summary>
    ///     Base class for the network layer
    /// </summary>
    public abstract class LayerBase
    {
        /// <summary>
        ///     Gets or sets the parameters.
        /// </summary>
        /// <value>
        ///     The parameters of the layer.
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

                return (T) o;
            }

            return default(T);
        }

        protected void SetParam<T>(string name, T value)
        {
            Params[name] = value;
        }

        internal abstract Function ToFunction(Variable inputFunction);
    }
}