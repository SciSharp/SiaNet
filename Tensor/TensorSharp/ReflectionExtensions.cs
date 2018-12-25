// ***********************************************************************
// Assembly         : TensorSharp
// Author           : Community
// Created          : 12-09-2018
//
// Last Modified By : Deepak Battini
// Last Modified On : 11-25-2018
// ***********************************************************************
// <copyright file="ReflectionExtensions.cs" company="TensorSharp">
//     Copyright (c) . All rights reserved.
// </copyright>
// <summary></summary>
// ***********************************************************************
using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Text;

namespace TensorSharp
{
    /// <summary>
    /// Class AssemblyExtensions.
    /// </summary>
    public static class AssemblyExtensions
    {
        /// <summary>
        /// Typeses the with attribute.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="assembly">The assembly.</param>
        /// <param name="inherit">if set to <c>true</c> [inherit].</param>
        /// <returns>IEnumerable&lt;Tuple&lt;Type, IEnumerable&lt;T&gt;&gt;&gt;.</returns>
        public static IEnumerable<Tuple<Type, IEnumerable<T>>> TypesWithAttribute<T>(this Assembly assembly, bool inherit)
        {
            foreach (var type in assembly.GetTypes())
            {
                var attributes = type.GetCustomAttributes(typeof(T), inherit);
                if (attributes.Any())
                {
                    yield return Tuple.Create(type, attributes.Cast<T>());
                }
            }
        }
    }

    /// <summary>
    /// Class TypeExtensions.
    /// </summary>
    public static class TypeExtensions
    {
        /// <summary>
        /// Methodses the with attribute.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="type">The type.</param>
        /// <param name="inherit">if set to <c>true</c> [inherit].</param>
        /// <returns>IEnumerable&lt;Tuple&lt;MethodInfo, IEnumerable&lt;T&gt;&gt;&gt;.</returns>
        public static IEnumerable<Tuple<MethodInfo, IEnumerable<T>>> MethodsWithAttribute<T>(this Type type, bool inherit)
        {
            foreach (var method in type.GetMethods())
            {
                var attributes = method.GetCustomAttributes(typeof(T), inherit);
                if (attributes.Any())
                {
                    yield return Tuple.Create(method, attributes.Cast<T>());
                }
            }
        }
    }

    /// <summary>
    /// Class MethodExtensions.
    /// </summary>
    public static class MethodExtensions
    {
        /// <summary>
        /// Parameterses the with attribute.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="method">The method.</param>
        /// <param name="inherit">if set to <c>true</c> [inherit].</param>
        /// <returns>IEnumerable&lt;Tuple&lt;ParameterInfo, IEnumerable&lt;T&gt;&gt;&gt;.</returns>
        public static IEnumerable<Tuple<ParameterInfo, IEnumerable<T>>> ParametersWithAttribute<T>(this MethodInfo method, bool inherit)
        {
            foreach (var paramter in method.GetParameters())
            {
                var attributes = paramter.GetCustomAttributes(typeof(T), inherit);
                if (attributes.Any())
                {
                    yield return Tuple.Create(paramter, attributes.Cast<T>());
                }
            }
        }
    }
}
