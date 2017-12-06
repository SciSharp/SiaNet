using CNTK;
using SiaNet.Common;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SiaNet.Model.Initializers
{
    /// <summary>
    /// Base for the initializer
    /// </summary>
    public class BaseInitializer
    {
        internal string Name { get; set; }

        internal double Scale { get; set; }

        internal uint? Seed { get; set; }

        internal int? OutputRank { get; set; }

        internal int? FilterRank { get; set; }

        internal BaseInitializer(string name)
        {
            Name = name;
            Scale = 0.01;
        }

        internal BaseInitializer(string name, double scale)
            : this(name)
        {
            Scale = scale;
        }

        internal BaseInitializer(string name, double scale, int seed)
            : this(name, scale)
        {
            Seed = (uint)seed;
        }

        internal BaseInitializer(string name, double scale, int outputRank, int filterRank, int seed)
            : this(name, scale, seed)
        {
            OutputRank = outputRank;
            FilterRank = filterRank;
        }

        internal CNTKDictionary Get()
        {
            CNTKDictionary result = null;
            switch (Name.Trim().ToLower())
            {
                case OptInitializers.Uniform:
                    result = Seed.HasValue ? CNTKLib.UniformInitializer(Scale, Seed.Value) : CNTKLib.UniformInitializer(Scale);
                    break;
                case OptInitializers.Normal:
                    result = CNTKLib.NormalInitializer(Scale);
                    result = Seed.HasValue ? CNTKLib.NormalInitializer(Scale, OutputRank.Value, FilterRank.Value, Seed.Value) : CNTKLib.NormalInitializer(Scale);
                    break;
                case OptInitializers.TruncatedNormal:
                    result = Seed.HasValue ? CNTKLib.TruncatedNormalInitializer(Scale, Seed.Value) : CNTKLib.TruncatedNormalInitializer(Scale);
                    break;
                case OptInitializers.Zeros:
                    result = CNTKLib.ConstantInitializer(0);
                    break;
                case OptInitializers.Ones:
                    result = CNTKLib.ConstantInitializer(1);
                    break;
                case OptInitializers.Constant:
                    result = CNTKLib.ConstantInitializer(Scale);
                    break;
                case OptInitializers.Xavier:
                    if (Seed.HasValue && !OutputRank.HasValue)
                        throw new ArithmeticException("Missing rank value when Seed is defined is defined for Xavier Initializer");
                    result = Seed.HasValue ? CNTKLib.XavierInitializer(Scale, OutputRank.Value, FilterRank.Value, Seed.Value) : CNTKLib.XavierInitializer(Scale);
                    break;
                case OptInitializers.GlorotNormal:
                    if (Seed.HasValue && !OutputRank.HasValue)
                        throw new ArithmeticException("Missing rank value when Seed is defined is defined for Glorot Normal Initializer");
                    result = Seed.HasValue ? CNTKLib.GlorotNormalInitializer(Scale, OutputRank.Value, FilterRank.Value, Seed.Value) : CNTKLib.GlorotNormalInitializer(Scale);
                    break;
                case OptInitializers.GlorotUniform:
                    if (Seed.HasValue && !OutputRank.HasValue)
                        throw new ArithmeticException("Missing rank value when Seed is defined is defined for Glorot Uniform Initializer");
                    result = Seed.HasValue ? CNTKLib.GlorotUniformInitializer(Scale, OutputRank.Value, FilterRank.Value, Seed.Value) : CNTKLib.GlorotUniformInitializer(Scale);
                    break;
                case OptInitializers.HeNormal:
                    if (Seed.HasValue && !OutputRank.HasValue)
                        throw new ArithmeticException("Missing rank value when Seed is defined is defined for He Normal Initializer");
                    result = CNTKLib.HeNormalInitializer(Scale);
                    result = Seed.HasValue ? CNTKLib.HeNormalInitializer(Scale, OutputRank.Value, FilterRank.Value, Seed.Value) : CNTKLib.HeNormalInitializer(Scale);
                    break;
                case OptInitializers.HeUniform:
                    if (Seed.HasValue && !OutputRank.HasValue)
                        throw new ArithmeticException("Missing rank value when Seed is defined is defined for He Uniform Initializer");
                    result = Seed.HasValue ? CNTKLib.HeUniformInitializer(Scale, OutputRank.Value, FilterRank.Value, Seed.Value) : CNTKLib.HeUniformInitializer(Scale);
                    break;
                default:
                    break;
            }

            return result;
        }
    }

    /// <summary>
    /// 
    /// </summary>
    /// <seealso cref="SiaNet.Model.Initializers.BaseInitializer" />
    public class Uniform : BaseInitializer
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="Uniform"/> class.
        /// </summary>
        public Uniform()
            : base(OptInitializers.Uniform)
        {
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="Uniform"/> class.
        /// </summary>
        /// <param name="scale">The scale value for the generator tensors.</param>
        public Uniform(int scale)
            : base(OptInitializers.Uniform, scale)
        {
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="Uniform"/> class.
        /// </summary>
        /// <param name="scale">The scale value for the generator tensors.</param>
        /// <param name="seed">Used to seed the random generator.</param>
        public Uniform(double scale, int seed)
            : base(OptInitializers.Uniform, scale, seed)
        {
        }
    }

    /// <summary>
    /// 
    /// </summary>
    /// <seealso cref="SiaNet.Model.Initializers.BaseInitializer" />
    public class Normal : BaseInitializer
    {
        public Normal()
            : base(OptInitializers.Normal)
        {
        }

        public Normal(double scale)
            : base(OptInitializers.Normal, scale)
        {
        }

        public Normal(double scale, int seed)
            : base(OptInitializers.Normal, scale, seed)
        {
        }
    }

    /// <summary>
    /// Initializer that generates a truncated normal distribution.
    /// </summary>
    /// <seealso cref="SiaNet.Model.Initializers.BaseInitializer" />
    public class TruncatedNormal : BaseInitializer
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="TruncatedNormal"/> class.
        /// </summary>
        public TruncatedNormal()
            : base(OptInitializers.TruncatedNormal)
        {
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="TruncatedNormal"/> class.
        /// </summary>
        /// <param name="stdev"> Standard deviation of the random values to generate.</param>
        public TruncatedNormal(double stdev)
            : base(OptInitializers.TruncatedNormal, stdev)
        {
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="TruncatedNormal"/> class.
        /// </summary>
        /// <param name="scale">The scale value for the generator tensors</param>
        /// <param name="seed">Used to seed the random generator.</param>
        public TruncatedNormal(double stdev, int seed)
            : base(OptInitializers.TruncatedNormal, stdev, seed)
        {
        }
    }

    /// <summary>
    /// Initializer that generates tensors initialized to a constant value.
    /// </summary>
    /// <seealso cref="SiaNet.Model.Initializers.BaseInitializer" />
    public class Constant : BaseInitializer
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="Constant"/> class.
        /// </summary>
        /// <param name="value">The value of the generator tensors.</param>
        public Constant(double value)
            : base(OptInitializers.Constant, value)
        {
        }
    }

    /// <summary>
    /// Initializer that generates tensors initialized to 0.
    /// </summary>
    /// <seealso cref="SiaNet.Model.Initializers.BaseInitializer" />
    public class Zeros : BaseInitializer
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="Zeros"/> class.
        /// </summary>
        public Zeros()
            : base(OptInitializers.Zeros, 0)
        {
        }
    }

    /// <summary>
    /// Initializer that generates tensors initialized to 1.
    /// </summary>
    /// <seealso cref="SiaNet.Model.Initializers.BaseInitializer" />
    public class Ones : BaseInitializer
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="Ones"/> class.
        /// </summary>
        public Ones()
            : base(OptInitializers.Ones, 1)
        {
        }
    }

    /// <summary>
    /// This initializer is designed to keep the scale of gradients roughly the same in all layers.
    /// </summary>
    /// <seealso cref="SiaNet.Model.Initializers.BaseInitializer" />
    public class Xavier : BaseInitializer
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="Xavier"/> class.
        /// </summary>
        public Xavier()
            : base(OptInitializers.Xavier)
        {
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="Xavier"/> class.
        /// </summary>
        /// <param name="scale">The scale value for the generator tensors.</param>
        public Xavier(double scale)
            : base(OptInitializers.Xavier, scale)
        {
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="Xavier"/> class.
        /// </summary>
        /// <param name="scale">The scale value for the generator tensors.</param>
        /// <param name="outputRank">The output rank value.</param>
        /// <param name="filterRank">The filter rank value.</param>
        /// <param name="seed">Used to seed the random generator.</param>
        public Xavier(double scale, int seed, int outputRank = 2147483647, int filterRank = 2147483647)
            : base(OptInitializers.Xavier, scale, outputRank, filterRank, seed)
        {
        }
    }

    /// <summary>
    /// Glorot normal initializer, also called Xavier normal initializer. It draws samples from a truncated normal distribution centered on 0 with stddev = sqrt(2 / (fan_in + fan_out)) where fan_in is the number of input units in the weight tensor and  fan_out is the number of output units in the weight tensor.
    /// </summary>
    /// <seealso cref="SiaNet.Model.Initializers.BaseInitializer" />
    public class GlorotNormal : BaseInitializer
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="GlorotNormal"/> class.
        /// </summary>
        public GlorotNormal()
            : base(OptInitializers.GlorotNormal)
        {
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="GlorotNormal"/> class.
        /// </summary>
        /// <param name="scale">The scale value for the generator tensors.</param>
        public GlorotNormal(double scale)
            : base(OptInitializers.GlorotNormal, scale)
        {
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="GlorotNormal"/> class.
        /// </summary>
        /// <param name="scale">The scale value for the generator tensors.</param>
        /// <param name="outputRank">The output rank value.</param>
        /// <param name="filterRank">The filter rank value.</param>
        /// <param name="seed">Used to seed the random generator.</param>
        public GlorotNormal(double scale, int seed, int outputRank = 2147483647, int filterRank = 2147483647)
            : base(OptInitializers.GlorotNormal, scale, outputRank, filterRank, seed)
        {
        }
    }

    /// <summary>
    /// Glorot uniform initializer, also called Xavier uniform initializer. It draws samples from a uniform distribution within[-limit, limit] where limit is sqrt(6 / (fan_in + fan_out)) where fan_in is the number of input units in the weight tensor and fan_out is the number of output units in the weight tensor.
    /// </summary>
    /// <seealso cref="SiaNet.Model.Initializers.BaseInitializer" />
    public class GlorotUniform : BaseInitializer
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="GlorotUniform"/> class.
        /// </summary>
        public GlorotUniform()
            : base(OptInitializers.GlorotUniform)
        {
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="GlorotUniform"/> class.
        /// </summary>
        /// <param name="scale">The scale value for the generator tensors.</param>
        public GlorotUniform(double scale)
            : base(OptInitializers.GlorotUniform, scale)
        {
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="GlorotUniform"/> class.
        /// </summary>
        /// <param name="scale">The scale value for the generator tensors.</param>
        /// <param name="outputRank">The output rank value.</param>
        /// <param name="filterRank">The filter rank value.</param>
        /// <param name="seed">Used to seed the random generator.</param>
        public GlorotUniform(double scale, int seed, int outputRank = 2147483647, int filterRank = 2147483647)
            : base(OptInitializers.GlorotUniform, scale, outputRank, filterRank, seed)
        {
        }
    }

    /// <summary>
    /// He normal initializer. It draws samples from a truncated normal distribution centered on 0 with stddev = sqrt(2 / fan_in) where fan_in is the number of input units in the weight tensor.
    /// </summary>
    /// <seealso cref="SiaNet.Model.Initializers.BaseInitializer" />
    public class HeNormal : BaseInitializer
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="HeNormal"/> class.
        /// </summary>
        public HeNormal()
            : base(OptInitializers.HeNormal)
        {
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="HeNormal"/> class.
        /// </summary>
        /// <param name="scale">The scale value for the generator tensors.</param>
        public HeNormal(double scale)
            : base(OptInitializers.HeNormal, scale)
        {
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="HeNormal"/> class.
        /// </summary>
        /// <param name="scale">The scale value for the generator tensors.</param>
        /// <param name="outputRank">The output rank value.</param>
        /// <param name="filterRank">The filter rank value.</param>
        /// <param name="seed">Used to seed the random generator.</param>
        public HeNormal(double scale, int seed, int outputRank = 2147483647, int filterRank = 2147483647)
            : base(OptInitializers.HeNormal, scale, outputRank, filterRank, seed)
        {
        }
    }

    /// <summary>
    /// He uniform variance scaling initializer. It draws samples from a uniform distribution within[-limit, limit] where limit is sqrt(6 / fan_in) where fan_in is the number of input units in the weight tensor.
    /// </summary>
    /// <seealso cref="SiaNet.Model.Initializers.BaseInitializer" />
    public class HeUniform : BaseInitializer
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="HeUniform"/> class.
        /// </summary>
        public HeUniform()
            : base(OptInitializers.HeUniform)
        {
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="HeUniform"/> class.
        /// </summary>
        /// <param name="scale">The scale value for the generator tensors.</param>
        public HeUniform(double scale)
            : base(OptInitializers.HeUniform, scale)
        {
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="HeUniform"/> class.
        /// </summary>
        /// <param name="scale">The scale value for the generator tensors.</param>
        /// <param name="outputRank">The output rank value.</param>
        /// <param name="filterRank">The filter rank value.</param>
        /// <param name="seed">Used to seed the random generator.</param>
        public HeUniform(double scale, int seed, int outputRank = 2147483647, int filterRank = 2147483647)
            : base(OptInitializers.HeUniform, scale, outputRank, filterRank, seed)
        {
        }
    }
}
