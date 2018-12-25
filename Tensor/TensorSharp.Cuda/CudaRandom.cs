// ***********************************************************************
// Assembly         : TensorSharp.CUDA91
// Author           : Community
// Created          : 12-09-2018
//
// Last Modified By : Deepak Battini
// Last Modified On : 11-25-2018
// ***********************************************************************
// <copyright file="CudaRandom.cs" company="TensorSharp.CUDA91">
//     Copyright (c) . All rights reserved.
// </copyright>
// <summary></summary>
// ***********************************************************************
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using TensorSharp.Cpu;

namespace TensorSharp.CUDA
{
    /// <summary>
    /// Basic implementation of random ops for CUDA. All we do here is generate the tensors on the
    /// CPU then copy to the CUDA buffer. This is definitely not an optimal implementation.
    /// </summary>
    [OpsClass]
    public class CudaRandom
    {
        /// <summary>
        /// The cpu allocator
        /// </summary>
        private readonly CpuAllocator cpuAllocator;
        /// <summary>
        /// The cpu random
        /// </summary>
        private readonly CpuRandom cpuRandom;

        /// <summary>
        /// Initializes a new instance of the <see cref="CudaRandom"/> class.
        /// </summary>
        public CudaRandom()
        {
            this.cpuAllocator = new CpuAllocator();
            this.cpuRandom = new CpuRandom();
        }


        /// <summary>
        /// Uniforms the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="seed">The seed.</param>
        /// <param name="min">The minimum.</param>
        /// <param name="max">The maximum.</param>
        [RegisterOpStorageType("random_uniform", typeof(CudaStorage))]
        public void Uniform(Tensor result, int? seed, float min, float max)
        {
            using (var cpuCopy = new Tensor(cpuAllocator, result.ElementType, result.Sizes))
            {
                cpuRandom.Uniform(cpuCopy, seed, min, max);
                Ops.Copy(result, cpuCopy);
            }
        }

        /// <summary>
        /// Normals the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="seed">The seed.</param>
        /// <param name="mean">The mean.</param>
        /// <param name="stdv">The STDV.</param>
        [RegisterOpStorageType("random_normal", typeof(CudaStorage))]
        public void Normal(Tensor result, int? seed, float mean, float stdv)
        {
            using (var cpuCopy = new Tensor(cpuAllocator, result.ElementType, result.Sizes))
            {
                cpuRandom.Normal(cpuCopy, seed, mean, stdv);
                Ops.Copy(result, cpuCopy);
            }
        }

        /// <summary>
        /// Exponentials the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="seed">The seed.</param>
        /// <param name="lambda">The lambda.</param>
        [RegisterOpStorageType("random_exponential", typeof(CudaStorage))]
        public void Exponential(Tensor result, int? seed, float lambda)
        {
            using (var cpuCopy = new Tensor(cpuAllocator, result.ElementType, result.Sizes))
            {
                cpuRandom.Exponential(cpuCopy, seed, lambda);
                Ops.Copy(result, cpuCopy);
            }
        }

        /// <summary>
        /// Cauchies the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="seed">The seed.</param>
        /// <param name="median">The median.</param>
        /// <param name="sigma">The sigma.</param>
        [RegisterOpStorageType("random_cauchy", typeof(CudaStorage))]
        public void Cauchy(Tensor result, int? seed, float median, float sigma)
        {
            using (var cpuCopy = new Tensor(cpuAllocator, result.ElementType, result.Sizes))
            {
                cpuRandom.Cauchy(cpuCopy, seed, median, sigma);
                Ops.Copy(result, cpuCopy);
            }
        }

        /// <summary>
        /// Logs the normal.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="seed">The seed.</param>
        /// <param name="mean">The mean.</param>
        /// <param name="stdv">The STDV.</param>
        [RegisterOpStorageType("random_lognormal", typeof(CudaStorage))]
        public void LogNormal(Tensor result, int? seed, float mean, float stdv)
        {
            using (var cpuCopy = new Tensor(cpuAllocator, result.ElementType, result.Sizes))
            {
                cpuRandom.LogNormal(cpuCopy, seed, mean, stdv);
                Ops.Copy(result, cpuCopy);
            }
        }

        /// <summary>
        /// Geometrics the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="seed">The seed.</param>
        /// <param name="p">The p.</param>
        [RegisterOpStorageType("random_geometric", typeof(CudaStorage))]
        public void Geometric(Tensor result, int? seed, float p)
        {
            using (var cpuCopy = new Tensor(cpuAllocator, result.ElementType, result.Sizes))
            {
                cpuRandom.Geometric(cpuCopy, seed, p);
                Ops.Copy(result, cpuCopy);
            }
        }

        /// <summary>
        /// Bernoullis the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="seed">The seed.</param>
        /// <param name="p">The p.</param>
        [RegisterOpStorageType("random_bernoulli", typeof(CudaStorage))]
        public void Bernoulli(Tensor result, int? seed, float p)
        {
            using (var cpuCopy = new Tensor(cpuAllocator, result.ElementType, result.Sizes))
            {
                cpuRandom.Bernoulli(cpuCopy, seed, p);
                Ops.Copy(result, cpuCopy);
            }
        }
    }
}
