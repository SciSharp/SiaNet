using System;
using CNTK;

namespace SiaNet.Initializers
{
    /// <summary>
    ///     This initializer is designed to keep the scale of gradients roughly the same in all layers.
    /// </summary>
    /// <seealso cref="InitializerBase" />
    public class Xavier : InitializerBase
    {
        private int? _filterRank;
        private int? _outputRank;
        private uint? _seed;

        /// <summary>
        ///     Initializes a new instance of the <see cref="Xavier" /> class.
        /// </summary>
        public Xavier() : this(0.01)
        {
        }

        /// <summary>
        ///     Initializes a new instance of the <see cref="Xavier" /> class.
        /// </summary>
        /// <param name="scale">The scale value for the generator tensors.</param>
        public Xavier(double scale)
        {
            Scale = scale;
        }

        /// <summary>
        ///     Initializes a new instance of the <see cref="Xavier" /> class.
        /// </summary>
        /// <param name="scale">The scale value for the generator tensors.</param>
        /// <param name="outputRank">The output rank value.</param>
        public Xavier(double scale, int outputRank) : this(scale)
        {
            OutputRank = outputRank;
        }

        /// <summary>
        ///     Initializes a new instance of the <see cref="Xavier" /> class.
        /// </summary>
        /// <param name="scale">The scale value for the generator tensors.</param>
        /// <param name="outputRank">The output rank value.</param>
        /// <param name="filterRank">The filter rank value.</param>
        public Xavier(double scale, int outputRank, int filterRank) : this(scale, outputRank)
        {
            FilterRank = filterRank;
        }

        /// <summary>
        ///     Initializes a new instance of the <see cref="Xavier" /> class.
        /// </summary>
        /// <param name="scale">The scale value for the generator tensors.</param>
        /// <param name="outputRank">The output rank value.</param>
        /// <param name="filterRank">The filter rank value.</param>
        /// <param name="seed">Used to seed the random generator.</param>
        public Xavier(double scale, int outputRank, int filterRank, uint seed) : this(scale, outputRank, filterRank)
        {
            Seed = seed;
        }

        public int FilterRank
        {
            get
            {
                if (!_filterRank.HasValue)
                {
                    throw new InvalidOperationException();
                }

                return _filterRank.Value;
            }
            protected set => _filterRank = value;
        }

        public bool HasFilterRank
        {
            get => _filterRank.HasValue;
        }

        public bool HasOutputRank
        {
            get => _outputRank.HasValue;
        }

        public bool HasSeed
        {
            get => _seed.HasValue;
        }

        public int OutputRank
        {
            get
            {
                if (!_outputRank.HasValue)
                {
                    throw new InvalidOperationException();
                }

                return _outputRank.Value;
            }
            protected set => _outputRank = value;
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
            if (HasOutputRank)
            {
                if (HasFilterRank)
                {
                    if (HasSeed)
                    {
                        return CNTKLib.XavierInitializer(Scale, OutputRank, FilterRank, Seed);
                    }

                    return CNTKLib.XavierInitializer(Scale, OutputRank, FilterRank);
                }

                return CNTKLib.XavierInitializer(Scale, OutputRank);
            }

            return CNTKLib.XavierInitializer(Scale);
        }
    }
}