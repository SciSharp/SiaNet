// ***********************************************************************
// Assembly         : TensorSharp
// Author           : Community
// Created          : 12-09-2018
//
// Last Modified By : Deepak Battini
// Last Modified On : 11-25-2018
// ***********************************************************************
// <copyright file="CpuRandom.cs" company="TensorSharp">
//     Copyright (c) . All rights reserved.
// </copyright>
// <summary></summary>
// ***********************************************************************
using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Text;

namespace TensorSharp.Cpu
{
    /// <summary>
    /// Class CpuRandom.
    /// </summary>
    [OpsClass]
    public class CpuRandom
    {
        /// <summary>
        /// The seed gen
        /// </summary>
        private static readonly Random seedGen = new Random();


        /// <summary>
        /// Initializes a new instance of the <see cref="CpuRandom"/> class.
        /// </summary>
        public CpuRandom()
        {
        }


        // allArgs should start with a null placeholder for the RNG object
        /// <summary>
        /// Invokes the with RNG.
        /// </summary>
        /// <param name="seed">The seed.</param>
        /// <param name="method">The method.</param>
        /// <param name="allArgs">All arguments.</param>
        private static void InvokeWithRng(int? seed, MethodInfo method, params object[] allArgs)
        {
            if (!seed.HasValue)
                seed = seedGen.Next();

            IntPtr rng;
            NativeWrapper.CheckResult(CpuOpsNative.TS_NewRNG(out rng));
            NativeWrapper.CheckResult(CpuOpsNative.TS_SetRNGSeed(rng, seed.Value));
            allArgs[0] = rng;
            NativeWrapper.InvokeTypeMatch(method, allArgs);
            NativeWrapper.CheckResult(CpuOpsNative.TS_DeleteRNG(rng));
        }

        /// <summary>
        /// The uniform function
        /// </summary>
        private MethodInfo uniform_func = NativeWrapper.GetMethod("TS_RandomUniform");
        /// <summary>
        /// Uniforms the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="seed">The seed.</param>
        /// <param name="min">The minimum.</param>
        /// <param name="max">The maximum.</param>
        [RegisterOpStorageType("random_uniform", typeof(CpuStorage))]
        public void Uniform(Tensor result, int? seed, float min, float max) { InvokeWithRng(seed, uniform_func, null, result, min, max); }

        /// <summary>
        /// The normal function
        /// </summary>
        private MethodInfo normal_func = NativeWrapper.GetMethod("TS_RandomNormal");
        /// <summary>
        /// Normals the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="seed">The seed.</param>
        /// <param name="mean">The mean.</param>
        /// <param name="stdv">The STDV.</param>
        [RegisterOpStorageType("random_normal", typeof(CpuStorage))]
        public void Normal(Tensor result, int? seed, float mean, float stdv) { InvokeWithRng(seed, normal_func, null, result, mean, stdv); }

        /// <summary>
        /// The exponential function
        /// </summary>
        private MethodInfo exponential_func = NativeWrapper.GetMethod("TS_RandomExponential");
        /// <summary>
        /// Exponentials the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="seed">The seed.</param>
        /// <param name="lambda">The lambda.</param>
        [RegisterOpStorageType("random_exponential", typeof(CpuStorage))]
        public void Exponential(Tensor result, int? seed, float lambda) { InvokeWithRng(seed, exponential_func, null, result, lambda); }

        /// <summary>
        /// The cauchy function
        /// </summary>
        private MethodInfo cauchy_func = NativeWrapper.GetMethod("TS_RandomCauchy");
        /// <summary>
        /// Cauchies the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="seed">The seed.</param>
        /// <param name="median">The median.</param>
        /// <param name="sigma">The sigma.</param>
        [RegisterOpStorageType("random_cauchy", typeof(CpuStorage))]
        public void Cauchy(Tensor result, int? seed, float median, float sigma) { InvokeWithRng(seed, cauchy_func, null, result, median, sigma); }

        /// <summary>
        /// The log normal function
        /// </summary>
        private MethodInfo log_normal_func = NativeWrapper.GetMethod("TS_RandomLogNormal");
        /// <summary>
        /// Logs the normal.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="seed">The seed.</param>
        /// <param name="mean">The mean.</param>
        /// <param name="stdv">The STDV.</param>
        [RegisterOpStorageType("random_lognormal", typeof(CpuStorage))]
        public void LogNormal(Tensor result, int? seed, float mean, float stdv) { InvokeWithRng(seed, log_normal_func, null, result, mean, stdv); }

        /// <summary>
        /// The geometric function
        /// </summary>
        private MethodInfo geometric_func = NativeWrapper.GetMethod("TS_RandomGeometric");
        /// <summary>
        /// Geometrics the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="seed">The seed.</param>
        /// <param name="p">The p.</param>
        [RegisterOpStorageType("random_geometric", typeof(CpuStorage))]
        public void Geometric(Tensor result, int? seed, float p) { InvokeWithRng(seed, geometric_func, null, result, p); }

        /// <summary>
        /// The bernoulli function
        /// </summary>
        private MethodInfo bernoulli_func = NativeWrapper.GetMethod("TS_RandomBernoulli");
        /// <summary>
        /// Bernoullis the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="seed">The seed.</param>
        /// <param name="p">The p.</param>
        [RegisterOpStorageType("random_bernoulli", typeof(CpuStorage))]
        public void Bernoulli(Tensor result, int? seed, float p) { InvokeWithRng(seed, bernoulli_func, null, result, p); }
    }
}
