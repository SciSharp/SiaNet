// ***********************************************************************
// Assembly         : SiaNet.Backend.TensorSharp
// Author           : Community
// Created          : 12-09-2018
//
// Last Modified By : Deepak Battini
// Last Modified On : 11-25-2018
// ***********************************************************************
// <copyright file="OpConstraint.cs" company="SiaNet.Backend.TensorSharp">
//     Copyright (c) . All rights reserved.
// </copyright>
// <summary></summary>
// ***********************************************************************
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace SiaNet.Backend.TensorSharp
{
    /// <summary>
    /// Class OpConstraint.
    /// </summary>
    public abstract class OpConstraint
    {
        /// <summary>
        /// Satisfieds for.
        /// </summary>
        /// <param name="args">The arguments.</param>
        /// <returns><c>true</c> if XXXX, <c>false</c> otherwise.</returns>
        public abstract bool SatisfiedFor(object[] args);
    }

    /// <summary>
    /// Class ArgCountConstraint.
    /// Implements the <see cref="SiaNet.Backend.TensorSharp.OpConstraint" />
    /// </summary>
    /// <seealso cref="SiaNet.Backend.TensorSharp.OpConstraint" />
    public class ArgCountConstraint : OpConstraint
    {
        /// <summary>
        /// The argument count
        /// </summary>
        private readonly int argCount;

        /// <summary>
        /// Initializes a new instance of the <see cref="ArgCountConstraint"/> class.
        /// </summary>
        /// <param name="argCount">The argument count.</param>
        public ArgCountConstraint(int argCount) { this.argCount = argCount; }

        /// <summary>
        /// Satisfieds for.
        /// </summary>
        /// <param name="args">The arguments.</param>
        /// <returns><c>true</c> if XXXX, <c>false</c> otherwise.</returns>
        public override bool SatisfiedFor(object[] args)
        {
            return args.Length == argCount;
        }
    }

    /// <summary>
    /// Class ArgTypeConstraint.
    /// Implements the <see cref="SiaNet.Backend.TensorSharp.OpConstraint" />
    /// </summary>
    /// <seealso cref="SiaNet.Backend.TensorSharp.OpConstraint" />
    public class ArgTypeConstraint : OpConstraint
    {
        /// <summary>
        /// The argument index
        /// </summary>
        private readonly int argIndex;
        /// <summary>
        /// The required type
        /// </summary>
        private readonly Type requiredType;

        /// <summary>
        /// Initializes a new instance of the <see cref="ArgTypeConstraint"/> class.
        /// </summary>
        /// <param name="argIndex">Index of the argument.</param>
        /// <param name="requiredType">Type of the required.</param>
        public ArgTypeConstraint(int argIndex, Type requiredType)
        {
            this.argIndex = argIndex;
            this.requiredType = requiredType;
        }

        /// <summary>
        /// Satisfieds for.
        /// </summary>
        /// <param name="args">The arguments.</param>
        /// <returns><c>true</c> if XXXX, <c>false</c> otherwise.</returns>
        public override bool SatisfiedFor(object[] args)
        {
            return requiredType.IsAssignableFrom(args[argIndex].GetType());
        }
    }

    /// <summary>
    /// Class ArgStorageTypeConstraint.
    /// Implements the <see cref="SiaNet.Backend.TensorSharp.OpConstraint" />
    /// </summary>
    /// <seealso cref="SiaNet.Backend.TensorSharp.OpConstraint" />
    public class ArgStorageTypeConstraint : OpConstraint
    {
        /// <summary>
        /// The argument index
        /// </summary>
        private readonly int argIndex;
        /// <summary>
        /// The required type
        /// </summary>
        private readonly Type requiredType;
        /// <summary>
        /// The allow null
        /// </summary>
        private readonly bool allowNull;

        /// <summary>
        /// Initializes a new instance of the <see cref="ArgStorageTypeConstraint"/> class.
        /// </summary>
        /// <param name="argIndex">Index of the argument.</param>
        /// <param name="requiredType">Type of the required.</param>
        /// <param name="allowNull">if set to <c>true</c> [allow null].</param>
        public ArgStorageTypeConstraint(int argIndex, Type requiredType, bool allowNull = true)
        {
            this.argIndex = argIndex;
            this.requiredType = requiredType;
            this.allowNull = allowNull;
        }

        /// <summary>
        /// Satisfieds for.
        /// </summary>
        /// <param name="args">The arguments.</param>
        /// <returns><c>true</c> if XXXX, <c>false</c> otherwise.</returns>
        public override bool SatisfiedFor(object[] args)
        {
            if (allowNull && args[argIndex] == null)
                return true;
            else if (!allowNull && args[argIndex] == null)
                return false;

            var argStorage = ((NDArray)args[argIndex]).Storage;
            return requiredType.IsAssignableFrom(argStorage.GetType());
        }
    }
}
