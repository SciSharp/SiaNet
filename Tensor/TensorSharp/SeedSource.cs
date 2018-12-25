// ***********************************************************************
// Assembly         : TensorSharp
// Author           : Community
// Created          : 12-09-2018
//
// Last Modified By : Deepak Battini
// Last Modified On : 11-25-2018
// ***********************************************************************
// <copyright file="SeedSource.cs" company="TensorSharp">
//     Copyright (c) . All rights reserved.
// </copyright>
// <summary></summary>
// ***********************************************************************
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace TensorSharp
{
    /// <summary>
    /// Class SeedSource.
    /// </summary>
    public class SeedSource
    {
        /// <summary>
        /// The RNG
        /// </summary>
        private Random rng;

        /// <summary>
        /// Initializes a new instance of the <see cref="SeedSource"/> class.
        /// </summary>
        public SeedSource()
        {
            rng = new Random();
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="SeedSource"/> class.
        /// </summary>
        /// <param name="seed">The seed.</param>
        public SeedSource(int seed)
        {
            rng = new Random();
        }

        /// <summary>
        /// Sets the seed.
        /// </summary>
        /// <param name="seed">The seed.</param>
        public void SetSeed(int seed)
        {
            rng = new Random(seed);
        }

        /// <summary>
        /// Nexts the seed.
        /// </summary>
        /// <returns>System.Int32.</returns>
        public int NextSeed()
        {
            return rng.Next();
        }
    }
}
