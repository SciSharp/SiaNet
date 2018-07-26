using System;
using CNTK;

namespace SiaNet.Model.Initializers
{
    /// <summary>
    ///     Glorot normal initializer, also called Xavier normal initializer. It draws samples from a truncated normal
    ///     distribution centered on 0 with stddev = sqrt(2 / (fan_in + fan_out)) where fan_in is the number of input units in
    ///     the weight tensor and  fan_out is the number of output units in the weight tensor.
    /// </summary>
    /// <seealso cref="InitializerBase" />
    public class GlorotNormal : InitializerBase
    {
        private int? _filterRank;
        private int? _outputRank;

        private uint? _seed;

        /// <summary>
        ///     Initializes a new instance of the <see cref="GlorotNormal" /> class.
        /// </summary>
        /// <param name="scale">The scale value for the generator tensors.</param>
        public GlorotNormal(double scale)
        {
            Scale = scale;
        }

        /// <summary>
        ///     Initializes a new instance of the <see cref="GlorotNormal" /> class.
        /// </summary>
        /// <param name="scale">The scale value for the generator tensors.</param>
        /// <param name="outputRank">The output rank value.</param>
        public GlorotNormal(double scale, int outputRank) : this(scale)
        {
            OutputRank = outputRank;
        }

        /// <summary>
        ///     Initializes a new instance of the <see cref="GlorotNormal" /> class.
        /// </summary>
        /// <param name="scale">The scale value for the generator tensors.</param>
        /// <param name="outputRank">The output rank value.</param>
        /// <param name="filterRank">The filter rank value.</param>
        public GlorotNormal(double scale, int outputRank, int filterRank) : this(scale, outputRank)
        {
            FilterRank = filterRank;
        }

        /// <summary>
        ///     Initializes a new instance of the <see cref="GlorotNormal" /> class.
        /// </summary>
        /// <param name="scale">The scale value for the generator tensors.</param>
        /// <param name="outputRank">The output rank value.</param>
        /// <param name="filterRank">The filter rank value.</param>
        /// <param name="seed">Used to seed the random generator.</param>
        public GlorotNormal(double scale, int outputRank, int filterRank, uint seed) : this(scale, outputRank,
            filterRank)
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
                        return CNTKLib.GlorotNormalInitializer(Scale, OutputRank, FilterRank, Seed);
                    }

                    return CNTKLib.GlorotNormalInitializer(Scale, OutputRank, FilterRank);
                }

                return CNTKLib.GlorotNormalInitializer(Scale, OutputRank);
            }

            return CNTKLib.GlorotNormalInitializer(Scale);
        }
    }
}