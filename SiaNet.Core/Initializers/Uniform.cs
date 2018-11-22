using System;
using CNTK;

namespace SiaNet.Initializers
{
    /// <summary>
    /// </summary>
    /// <seealso cref="InitializerBase" />
    public class Uniform : InitializerBase
    {
        private uint? _seed;

        /// <summary>
        ///     Initializes a new instance of the <see cref="Uniform" /> class.
        /// </summary>
        public Uniform() : this(0.01)
        {
        }

        /// <summary>
        ///     Initializes a new instance of the <see cref="Uniform" /> class.
        /// </summary>
        /// <param name="scale">The scale value for the generator tensors.</param>
        public Uniform(double scale)
        {
            Scale = scale;
        }

        /// <summary>
        ///     Initializes a new instance of the <see cref="Uniform" /> class.
        /// </summary>
        /// <param name="scale">The scale value for the generator tensors.</param>
        /// <param name="seed">Used to seed the random generator.</param>
        public Uniform(double scale, uint seed) : this(scale)
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
            return HasSeed ? CNTKLib.UniformInitializer(Scale, Seed) : CNTKLib.UniformInitializer(Scale);
        }
    }
}