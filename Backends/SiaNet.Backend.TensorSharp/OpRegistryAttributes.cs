// ***********************************************************************
// Assembly         : SiaNet.Backend.TensorSharp
// Author           : Community
// Created          : 12-09-2018
//
// Last Modified By : Deepak Battini
// Last Modified On : 11-25-2018
// ***********************************************************************
// <copyright file="OpRegistryAttributes.cs" company="SiaNet.Backend.TensorSharp">
//     Copyright (c) . All rights reserved.
// </copyright>
// <summary></summary>
// ***********************************************************************
using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Text;

namespace SiaNet.Backend.TensorSharp
{
    /// <summary>
    /// Class OpsClassAttribute.
    /// Implements the <see cref="System.Attribute" />
    /// </summary>
    /// <seealso cref="System.Attribute" />
    [AttributeUsage(AttributeTargets.Class)]
    public class OpsClassAttribute : Attribute
    {
    }

    /// <summary>
    /// Class RegisterOp.
    /// Implements the <see cref="System.Attribute" />
    /// </summary>
    /// <seealso cref="System.Attribute" />
    [AttributeUsage(AttributeTargets.Method)]
    public abstract class RegisterOp : Attribute
    {
        /// <summary>
        /// Gets the name of the op.
        /// </summary>
        /// <value>The name of the op.</value>
        public string OpName { get; private set; }

        /// <summary>
        /// Initializes a new instance of the <see cref="RegisterOp"/> class.
        /// </summary>
        /// <param name="opName">Name of the op.</param>
        public RegisterOp(string opName)
        {
            this.OpName = opName;
        }

        /// <summary>
        /// Does the register.
        /// </summary>
        /// <param name="instance">The instance.</param>
        /// <param name="method">The method.</param>
        /// <param name="paramConstraints">The parameter constraints.</param>
        public abstract void DoRegister(object instance, MethodInfo method, IEnumerable<OpConstraint> paramConstraints);
    }

    /// <summary>
    /// Register a method where the only constraint is that the argument counts match.
    /// Implements the <see cref="SiaNet.Backend.TensorSharp.RegisterOp" />
    /// </summary>
    /// <seealso cref="SiaNet.Backend.TensorSharp.RegisterOp" />
    [AttributeUsage(AttributeTargets.Method)]
    public class RegisterOpArgCount : RegisterOp
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="RegisterOpArgCount"/> class.
        /// </summary>
        /// <param name="opName">Name of the op.</param>
        public RegisterOpArgCount(string opName) : base(opName)
        {
        }

        /// <summary>
        /// Does the register.
        /// </summary>
        /// <param name="instance">The instance.</param>
        /// <param name="method">The method.</param>
        /// <param name="paramConstraints">The parameter constraints.</param>
        public override void DoRegister(object instance, MethodInfo method, IEnumerable<OpConstraint> paramConstraints)
        {
            var constraints = new List<OpConstraint>();
            constraints.AddRange(paramConstraints);
            constraints.Add(new ArgCountConstraint(method.GetParameters().Length));

            OpRegistry.Register(OpName, args => method.Invoke(instance, args), constraints);
        }
    }


    /// <summary>
    /// Class RegisterOpStorageType.
    /// Implements the <see cref="SiaNet.Backend.TensorSharp.RegisterOp" />
    /// </summary>
    /// <seealso cref="SiaNet.Backend.TensorSharp.RegisterOp" />
    [AttributeUsage(AttributeTargets.Method)]
    public class RegisterOpStorageType : RegisterOp
    {
        /// <summary>
        /// The storage type
        /// </summary>
        private readonly Type storageType;

        /// <summary>
        /// Initializes a new instance of the <see cref="RegisterOpStorageType"/> class.
        /// </summary>
        /// <param name="opName">Name of the op.</param>
        /// <param name="storageType">Type of the storage.</param>
        public RegisterOpStorageType(string opName, Type storageType) : base(opName)
        {
            this.storageType = storageType;
        }

        /// <summary>
        /// Does the register.
        /// </summary>
        /// <param name="instance">The instance.</param>
        /// <param name="method">The method.</param>
        /// <param name="paramConstraints">The parameter constraints.</param>
        public override void DoRegister(object instance, MethodInfo method, IEnumerable<OpConstraint> paramConstraints)
        {
            var constraints = new List<OpConstraint>();
            constraints.AddRange(paramConstraints);
            constraints.Add(new ArgCountConstraint(method.GetParameters().Length));

            var methodParams = method.GetParameters();
            for(int i = 0; i < methodParams.Length; ++i)
            {
                if (methodParams[i].ParameterType == typeof(NDArray))
                {
                    constraints.Add(new ArgStorageTypeConstraint(i, storageType));
                }
            }

            OpRegistry.Register(OpName, args => method.Invoke(instance, args), constraints);
        }
    }




    /// <summary>
    /// Class ArgConstraintAttribute.
    /// Implements the <see cref="System.Attribute" />
    /// </summary>
    /// <seealso cref="System.Attribute" />
    [AttributeUsage(AttributeTargets.Parameter)]
    public abstract class ArgConstraintAttribute : Attribute
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="ArgConstraintAttribute"/> class.
        /// </summary>
        public ArgConstraintAttribute()
        {
        }

        /// <summary>
        /// Gets the constraints.
        /// </summary>
        /// <param name="parameter">The parameter.</param>
        /// <param name="instance">The instance.</param>
        /// <returns>IEnumerable&lt;OpConstraint&gt;.</returns>
        public abstract IEnumerable<OpConstraint> GetConstraints(ParameterInfo parameter, object instance);
    }

    /// <summary>
    /// Class OpArgStorageType.
    /// Implements the <see cref="SiaNet.Backend.TensorSharp.ArgConstraintAttribute" />
    /// </summary>
    /// <seealso cref="SiaNet.Backend.TensorSharp.ArgConstraintAttribute" />
    [AttributeUsage(AttributeTargets.Parameter)]
    public class OpArgStorageType : ArgConstraintAttribute
    {
        /// <summary>
        /// The storage type
        /// </summary>
        private readonly Type storageType;

        /// <summary>
        /// Initializes a new instance of the <see cref="OpArgStorageType"/> class.
        /// </summary>
        /// <param name="storageType">Type of the storage.</param>
        public OpArgStorageType(Type storageType)
        {
            this.storageType = storageType;
        }

        /// <summary>
        /// Gets the constraints.
        /// </summary>
        /// <param name="parameter">The parameter.</param>
        /// <param name="instance">The instance.</param>
        /// <returns>IEnumerable&lt;OpConstraint&gt;.</returns>
        public override IEnumerable<OpConstraint> GetConstraints(ParameterInfo parameter, object instance)
        {
            yield return new ArgStorageTypeConstraint(parameter.Position, storageType);
        }
    }
}
