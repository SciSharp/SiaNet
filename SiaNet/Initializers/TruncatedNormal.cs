using System;
using CNTK;

namespace SiaNet.Initializers
{
    /// <summary>
    ///     Initializer that generates a truncated normal distribution.
    /// </summary>
    /// <seealso cref="InitializerBase" />
    public class TruncatedNormal : InitializerBase
    {
        private uint? _seed;

        /// <summary>
        ///     Initializes a new instance of the <see cref="TruncatedNormal" /> class.
        /// </summary>
        public TruncatedNormal() : this(0.01)
        {
        }

        /// <summary>
        ///     Initializes a new instance of the <see cref="TruncatedNormal" /> class.
        /// </summary>
        /// <param name="scale">Standard deviation of the random values to generate.</param>
        public TruncatedNormal(double scale)
        {
            Scale = scale;
        }

        /// <summary>
        ///     Initializes a new instance of the <see cref="TruncatedNormal" /> class.
        /// </summary>
        /// <param name="scale">Standard deviation of the random values to generate.</param>
        /// <param name="seed">Used to seed the random generator.</param>
        public TruncatedNormal(double scale, uint seed) : this(scale)
        {
            Seed = seed;
        }

        public bool HasSeed
        {
            get => _seed.HasValue;
        }

        public double Scale { get; protected set; }

        public uint Seed
        {
            get
            {
                if (!_seed.HasValue)
                {
                    throw new InvalidOperationException();
                }

                return _seed.Value;
            }
            protected set => _seed = value;
        }

        /// <inheritdoc />
        internal override CNTKDictionary ToDictionary()
        {
            return HasSeed
                ? CNTKLib.TruncatedNormalInitializer(Scale, Seed)
                : CNTKLib.TruncatedNormalInitializer(Scale);
        }
    }
}