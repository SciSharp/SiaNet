// ***********************************************************************
// Assembly         : TensorSharp
// Author           : Community
// Created          : 12-09-2018
//
// Last Modified By : Deepak Battini
// Last Modified On : 11-25-2018
// ***********************************************************************
// <copyright file="OpRegistry.cs" company="TensorSharp">
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
    /// Delegate OpHandler
    /// </summary>
    /// <param name="args">The arguments.</param>
    /// <returns>System.Object.</returns>
    public delegate object OpHandler(object[] args);

    /// <summary>
    /// Class OpRegistry.
    /// </summary>
    public static class OpRegistry
    {
        /// <summary>
        /// Class OpInstance.
        /// </summary>
        private class OpInstance
        {
            /// <summary>
            /// The handler
            /// </summary>
            public OpHandler handler;
            /// <summary>
            /// The constraints
            /// </summary>
            public IEnumerable<OpConstraint> constraints;
        }

        /// <summary>
        /// The op instances
        /// </summary>
        private static Dictionary<string, List<OpInstance>> opInstances = new Dictionary<string, List<OpInstance>>();
        // Remember which assemblies have been registered to avoid accidental double-registering
        /// <summary>
        /// The registered assemblies
        /// </summary>
        private static HashSet<Assembly> registeredAssemblies = new HashSet<Assembly>();

        /// <summary>
        /// Initializes static members of the <see cref="OpRegistry"/> class.
        /// </summary>
        static OpRegistry()
        {
            // Register CPU ops from this assembly
            RegisterAssembly(Assembly.GetExecutingAssembly());
        }

        /// <summary>
        /// Registers the specified op name.
        /// </summary>
        /// <param name="opName">Name of the op.</param>
        /// <param name="handler">The handler.</param>
        /// <param name="constraints">The constraints.</param>
        public static void Register(string opName, OpHandler handler, IEnumerable<OpConstraint> constraints)
        {
            var newInstance = new OpInstance() { handler = handler, constraints = constraints };

            List<OpInstance> instanceList;
            if (opInstances.TryGetValue(opName, out instanceList))
            {
                instanceList.Add(newInstance);
            }
            else
            {
                instanceList = new List<OpInstance>();
                instanceList.Add(newInstance);
                opInstances.Add(opName, instanceList);
            }
        }

        /// <summary>
        /// Invokes the specified op name.
        /// </summary>
        /// <param name="opName">Name of the op.</param>
        /// <param name="args">The arguments.</param>
        /// <returns>System.Object.</returns>
        /// <exception cref="ApplicationException">
        /// None of the registered handlers match the arguments for " + opName
        /// or
        /// No handlers have been registered for op " + opName
        /// </exception>
        public static object Invoke(string opName, params object[] args)
        {
            List<OpInstance> instanceList;
            if (opInstances.TryGetValue(opName, out instanceList))
            {
                foreach (var instance in instanceList)
                {
                    if (instance.constraints.All(x => x.SatisfiedFor(args)))
                    {
                        return instance.handler.Invoke(args);
                    }
                }

                throw new ApplicationException("None of the registered handlers match the arguments for " + opName);
            }
            else
            {
                throw new ApplicationException("No handlers have been registered for op " + opName);
            }
        }

        /// <summary>
        /// Registers the assembly.
        /// </summary>
        /// <param name="assembly">The assembly.</param>
        public static void RegisterAssembly(Assembly assembly)
        {
            if (!registeredAssemblies.Contains(assembly))
            {
                registeredAssemblies.Add(assembly);

                var types = assembly.TypesWithAttribute<OpsClassAttribute>(false)
                    .Select(x => x.Item1);

                foreach (var type in types)
                {
                    var instance = Activator.CreateInstance(type);

                    var methods = type.MethodsWithAttribute<RegisterOp>(false);
                    foreach (var method in methods)
                    {
                        var paramConstraints = GetParameterConstraints(method.Item1, instance);
                        foreach (var attribute in method.Item2)
                        {
                            attribute.DoRegister(instance, method.Item1, paramConstraints);
                        }
                    }
                }
            }
        }

        /// <summary>
        /// Gets the parameter constraints.
        /// </summary>
        /// <param name="method">The method.</param>
        /// <param name="instance">The instance.</param>
        /// <returns>IEnumerable&lt;OpConstraint&gt;.</returns>
        private static IEnumerable<OpConstraint> GetParameterConstraints(MethodInfo method, object instance)
        {
            var result = Enumerable.Empty<OpConstraint>();
            foreach (var parameter in method.ParametersWithAttribute<ArgConstraintAttribute>(false))
            {
                foreach (var attribute in parameter.Item2)
                {
                    result = Enumerable.Concat(result, attribute.GetConstraints(parameter.Item1, instance));
                }
            }

            return result;
        }
    }
}
