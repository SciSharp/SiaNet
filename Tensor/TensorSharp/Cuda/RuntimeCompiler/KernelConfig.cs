// ***********************************************************************
// Assembly         : TensorSharp.CUDA91
// Author           : Community
// Created          : 12-09-2018
//
// Last Modified By : Deepak Battini
// Last Modified On : 11-25-2018
// ***********************************************************************
// <copyright file="KernelConfig.cs" company="TensorSharp.CUDA91">
//     Copyright (c) . All rights reserved.
// </copyright>
// <summary></summary>
// ***********************************************************************
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace TensorSharp.CUDA.RuntimeCompiler
{
    /// <summary>
    /// Class KernelConfig.
    /// </summary>
    public class KernelConfig
    {
        /// <summary>
        /// The values
        /// </summary>
        private readonly SortedDictionary<string, string> values = new SortedDictionary<string, string>();


        /// <summary>
        /// Initializes a new instance of the <see cref="KernelConfig"/> class.
        /// </summary>
        public KernelConfig()
        {
        }

        /// <summary>
        /// Gets the keys.
        /// </summary>
        /// <value>The keys.</value>
        public IEnumerable<string> Keys { get { return values.Keys; } }

        /// <summary>
        /// Alls the values.
        /// </summary>
        /// <returns>IEnumerable&lt;KeyValuePair&lt;System.String, System.String&gt;&gt;.</returns>
        public IEnumerable<KeyValuePair<string, string>> AllValues()
        {
            return values;
        }

        /// <summary>
        /// Determines whether the specified <see cref="System.Object" /> is equal to this instance.
        /// </summary>
        /// <param name="obj">The object to compare with the current object.</param>
        /// <returns><c>true</c> if the specified <see cref="System.Object" /> is equal to this instance; otherwise, <c>false</c>.</returns>
        public override bool Equals(object obj)
        {
            var o = obj as KernelConfig;
            if (o == null) return false;

            if (values.Count != o.values.Count) return false;

            foreach (var kvp in values)
            {
                string oValue;
                if (values.TryGetValue(kvp.Key, out oValue))
                {
                    if (!kvp.Value.Equals(oValue))
                        return false;
                }
                else
                {
                    return false;
                }
            }

            return true;
        }

        /// <summary>
        /// Returns a hash code for this instance.
        /// </summary>
        /// <returns>A hash code for this instance, suitable for use in hashing algorithms and data structures like a hash table.</returns>
        public override int GetHashCode()
        {
            int result = 0;
            foreach (var kvp in values)
            {
                result ^= kvp.Key.GetHashCode();
                result ^= kvp.Value.GetHashCode();
            }
            return result;
        }

        /// <summary>
        /// Determines whether the specified name contains key.
        /// </summary>
        /// <param name="name">The name.</param>
        /// <returns><c>true</c> if the specified name contains key; otherwise, <c>false</c>.</returns>
        public bool ContainsKey(string name)
        {
            return values.ContainsKey(name);
        }

        /// <summary>
        /// Sets the specified name.
        /// </summary>
        /// <param name="name">The name.</param>
        /// <param name="value">The value.</param>
        public void Set(string name, string value)
        {
            values[name] = value;
        }

        /// <summary>
        /// Applies to template.
        /// </summary>
        /// <param name="templateCode">The template code.</param>
        /// <returns>System.String.</returns>
        public string ApplyToTemplate(string templateCode)
        {
            var fullCode = new StringBuilder();
            foreach (var item in values)
            {
                fullCode.AppendFormat("#define {0} {1}\n", item.Key, item.Value);
            }
            fullCode.AppendLine(templateCode);
            return fullCode.ToString();
        }
    }
}
