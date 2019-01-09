// ***********************************************************************
// Assembly         : TensorSharp.CUDA91
// Author           : Community
// Created          : 12-09-2018
//
// Last Modified By : Deepak Battini
// Last Modified On : 11-25-2018
// ***********************************************************************
// <copyright file="CudaIncludeAttribute.cs" company="TensorSharp.CUDA91">
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
    /// Class CudaIncludeAttribute.
    /// Implements the <see cref="System.Attribute" />
    /// </summary>
    /// <seealso cref="System.Attribute" />
    [AttributeUsage(AttributeTargets.Class, AllowMultiple = true, Inherited = false)]
    public class CudaIncludeAttribute : Attribute
    {
        /// <summary>
        /// Gets the name of the field.
        /// </summary>
        /// <value>The name of the field.</value>
        public string FieldName { get; private set; }
        /// <summary>
        /// Gets the name of the include.
        /// </summary>
        /// <value>The name of the include.</value>
        public string IncludeName { get; private set; }

        /// <summary>
        /// Initializes a new instance of the <see cref="CudaIncludeAttribute"/> class.
        /// </summary>
        /// <param name="fieldName">Name of the field.</param>
        /// <param name="includeName">Name of the include.</param>
        public CudaIncludeAttribute(string fieldName, string includeName)
        {
            this.FieldName = fieldName;
            this.IncludeName = includeName;
        }
    }
}
