// ***********************************************************************
// Assembly         : SiaNet.Backend.TensorSharp
// Author           : Community
// Created          : 12-09-2018
//
// Last Modified By : Deepak Battini
// Last Modified On : 12-09-2018
// ***********************************************************************
// <copyright file="DelegateDisposable.cs" company="SiaNet.Backend.TensorSharp">
//     Copyright (c) . All rights reserved.
// </copyright>
// <summary></summary>
// ***********************************************************************
namespace SiaNet.Backend.TensorSharp.Core
{
    using System;

    /// <summary>
    /// Delegate Disposable class
    /// Implements the <see cref="System.IDisposable" />
    /// </summary>
    /// <seealso cref="System.IDisposable" />
    public class DelegateDisposable : IDisposable
    {
        /// <summary>
        /// The action
        /// </summary>
        private readonly Action action;

        /// <summary>
        /// Initializes a new instance of the <see cref="DelegateDisposable"/> class.
        /// </summary>
        /// <param name="action">The action.</param>
        public DelegateDisposable(Action action)
        {
            this.action = action;
        }

        /// <summary>
        /// Performs application-defined tasks associated with freeing, releasing, or resetting unmanaged resources.
        /// </summary>
        public virtual void Dispose()
        {
            action();
        }
    }
}
