// ***********************************************************************
// Assembly         : TensorSharp.CUDA91
// Author           : Community
// Created          : 12-09-2018
//
// Last Modified By : Deepak Battini
// Last Modified On : 11-25-2018
// ***********************************************************************
// <copyright file="ObjectPool.cs" company="TensorSharp.CUDA91">
//     Copyright (c) . All rights reserved.
// </copyright>
// <summary></summary>
// ***********************************************************************
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace TensorSharp.CUDA.Util
{
    /// <summary>
    /// Class PooledObject.
    /// Implements the <see cref="System.IDisposable" />
    /// </summary>
    /// <typeparam name="T"></typeparam>
    /// <seealso cref="System.IDisposable" />
    public class PooledObject<T> : IDisposable
    {
        /// <summary>
        /// The on dispose
        /// </summary>
        private readonly Action<PooledObject<T>> onDispose;
        /// <summary>
        /// The value
        /// </summary>
        private readonly T value;

        /// <summary>
        /// The disposed
        /// </summary>
        private bool disposed = false;

        /// <summary>
        /// Initializes a new instance of the <see cref="PooledObject{T}"/> class.
        /// </summary>
        /// <param name="value">The value.</param>
        /// <param name="onDispose">The on dispose.</param>
        /// <exception cref="ArgumentNullException">onDispose</exception>
        public PooledObject(T value, Action<PooledObject<T>> onDispose)
        {
            if (onDispose == null) throw new ArgumentNullException("onDispose");

            this.onDispose = onDispose;
            this.value = value;
        }

        /// <summary>
        /// Gets the value.
        /// </summary>
        /// <value>The value.</value>
        /// <exception cref="ObjectDisposedException"></exception>
        public T Value
        {
            get
            {
                if (disposed) throw new ObjectDisposedException(this.ToString());
                return value;
            }
        }

        /// <summary>
        /// Performs application-defined tasks associated with freeing, releasing, or resetting unmanaged resources.
        /// </summary>
        /// <exception cref="ObjectDisposedException"></exception>
        public void Dispose()
        {
            if (!disposed)
            {
                onDispose(this);
                disposed = true;
            }
            else
            {
                throw new ObjectDisposedException(this.ToString());
            }
        }
    }

    /// <summary>
    /// Class ObjectPool.
    /// Implements the <see cref="System.IDisposable" />
    /// </summary>
    /// <typeparam name="T"></typeparam>
    /// <seealso cref="System.IDisposable" />
    public class ObjectPool<T> : IDisposable
    {
        /// <summary>
        /// The constructor
        /// </summary>
        private readonly Func<T> constructor;
        /// <summary>
        /// The destructor
        /// </summary>
        private readonly Action<T> destructor;
        /// <summary>
        /// The free list
        /// </summary>
        private readonly Stack<T> freeList = new Stack<T>();
        /// <summary>
        /// The disposed
        /// </summary>
        private bool disposed = false;


        /// <summary>
        /// Initializes a new instance of the <see cref="ObjectPool{T}"/> class.
        /// </summary>
        /// <param name="initialSize">The initial size.</param>
        /// <param name="constructor">The constructor.</param>
        /// <param name="destructor">The destructor.</param>
        /// <exception cref="ArgumentNullException">
        /// constructor
        /// or
        /// destructor
        /// </exception>
        public ObjectPool(int initialSize, Func<T> constructor, Action<T> destructor)
        {
            if (constructor == null) throw new ArgumentNullException("constructor");
            if (destructor == null) throw new ArgumentNullException("destructor");

            this.constructor = constructor;
            this.destructor = destructor;

            for(int i = 0; i < initialSize; ++i)
            {
                freeList.Push(constructor());
            }
        }

        /// <summary>
        /// Performs application-defined tasks associated with freeing, releasing, or resetting unmanaged resources.
        /// </summary>
        public void Dispose()
        {
            if (!disposed)
            {
                disposed = true;
                foreach (var item in freeList)
                {
                    destructor(item);
                }
                freeList.Clear();
            }
        }

        /// <summary>
        /// Gets this instance.
        /// </summary>
        /// <returns>PooledObject&lt;T&gt;.</returns>
        public PooledObject<T> Get()
        {
            T value = freeList.Count > 0 ? freeList.Pop() : constructor();
            return new PooledObject<T>(value, Release);
        }

        /// <summary>
        /// Releases the specified handle.
        /// </summary>
        /// <param name="handle">The handle.</param>
        private void Release(PooledObject<T> handle)
        {
            freeList.Push(handle.Value);
        }
    }
}
