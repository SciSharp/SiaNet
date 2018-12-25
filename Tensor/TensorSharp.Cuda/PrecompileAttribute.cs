// ***********************************************************************
// Assembly         : TensorSharp.CUDA91
// Author           : Community
// Created          : 12-09-2018
//
// Last Modified By : Deepak Battini
// Last Modified On : 11-25-2018
// ***********************************************************************
// <copyright file="PrecompileAttribute.cs" company="TensorSharp.CUDA91">
//     Copyright (c) . All rights reserved.
// </copyright>
// <summary></summary>
// ***********************************************************************
using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Text;
using TensorSharp.CUDA.RuntimeCompiler;

namespace TensorSharp.CUDA
{
    /// <summary>
    /// Class PrecompileAttribute.
    /// Implements the <see cref="System.Attribute" />
    /// </summary>
    /// <seealso cref="System.Attribute" />
    [AttributeUsage(AttributeTargets.Class, AllowMultiple =false, Inherited =false)]
    public class PrecompileAttribute : Attribute
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="PrecompileAttribute"/> class.
        /// </summary>
        public PrecompileAttribute()
        {
        }
    }

    /// <summary>
    /// Interface IPrecompilable
    /// </summary>
    public interface IPrecompilable
    {
        /// <summary>
        /// Precompiles the specified compiler.
        /// </summary>
        /// <param name="compiler">The compiler.</param>
        void Precompile(CudaCompiler compiler);
    }

    /// <summary>
    /// Class PrecompileHelper.
    /// </summary>
    public static class PrecompileHelper
    {
        /// <summary>
        /// Precompiles all fields.
        /// </summary>
        /// <param name="instance">The instance.</param>
        /// <param name="compiler">The compiler.</param>
        public static void PrecompileAllFields(object instance, CudaCompiler compiler)
        {
            var type = instance.GetType();

            foreach (var field in type.GetFields())
            {
                if (typeof(IPrecompilable).IsAssignableFrom(field.FieldType))
                {
                    var precompilableField = (IPrecompilable)field.GetValue(instance);
                    Console.WriteLine("Compiling field " + field.Name);
                    precompilableField.Precompile(compiler);
                }
            }
        }
    }
}
