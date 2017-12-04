namespace SiaNet
{
    using CNTK;
    using SiaNet.Common;
    using System;

    /// <summary>
    /// Initializations define the way to set the initial random weights of Keras layers.The following built-in initializers are available as part of the SiaNet.Common.OptInitializers module
    /// <see cref="OptInitializers"/>
    /// </summary>
    public class Initializers
    {
        /// <summary>
        /// Gets the specified initializers based on function name.
        /// </summary>
        /// <param name="initializers">The initializers name.</param>
        /// <param name="scale">The scale unit.</param>
        /// <param name="seed">The seed value.</param>
        /// <param name="rank">The rank. Tuple of output and filter rank in integer. Tuple.Create(2,1)</param>
        /// <returns>
        /// Initializer instance as CNTKDictionary.
        /// </returns>
        public static CNTKDictionary Get(string initializers, double scale = 0.1, uint? seed = null, Tuple<int, int> rank = null)
        {
            CNTKDictionary result = null;
            switch (initializers.Trim().ToLower())
            {
                case OptInitializers.Uniform:
                    result = seed.HasValue ? CNTKLib.UniformInitializer(scale, seed.Value) : CNTKLib.UniformInitializer(scale);
                    break;
                case OptInitializers.Normal:
                    if (seed.HasValue && rank == null)
                        throw new ArithmeticException("Missing rank value when seed is defined for Normal Initializer");

                    result = CNTKLib.NormalInitializer(scale);
                    result = seed.HasValue ? CNTKLib.NormalInitializer(scale, rank.Item1, rank.Item2, seed.Value) : CNTKLib.NormalInitializer(scale);
                    break;
                case OptInitializers.TruncatedNormal:
                    result = seed.HasValue ? CNTKLib.TruncatedNormalInitializer(scale, seed.Value) : CNTKLib.TruncatedNormalInitializer(scale);
                    break;
                case OptInitializers.Zeros:
                    result = CNTKLib.ConstantInitializer(0);
                    break;
                case OptInitializers.Ones:
                    result = CNTKLib.ConstantInitializer(1);
                    break;
                case OptInitializers.Constant:
                    result = CNTKLib.ConstantInitializer(scale);
                    break;
                case OptInitializers.Xavier:
                    if (seed.HasValue && rank == null)
                        throw new ArithmeticException("Missing rank value when seed is defined is defined for Xavier Initializer");
                    result = seed.HasValue ? CNTKLib.XavierInitializer(scale, rank.Item1, rank.Item2, seed.Value) : CNTKLib.XavierInitializer(scale);
                    break;
                case OptInitializers.GlorotNormal:
                    if (seed.HasValue && rank == null)
                        throw new ArithmeticException("Missing rank value when seed is defined is defined for Glorot Normal Initializer");
                    result = seed.HasValue ? CNTKLib.GlorotNormalInitializer(scale, rank.Item1, rank.Item2, seed.Value) : CNTKLib.GlorotNormalInitializer(scale);
                    break;
                case OptInitializers.GlorotUniform:
                    if (seed.HasValue && rank == null)
                        throw new ArithmeticException("Missing rank value when seed is defined is defined for Glorot Uniform Initializer");
                    result = seed.HasValue ? CNTKLib.GlorotUniformInitializer(scale, rank.Item1, rank.Item2, seed.Value) : CNTKLib.GlorotUniformInitializer(scale);
                    break;
                case OptInitializers.HeNormal:
                    if (seed.HasValue && rank == null)
                        throw new ArithmeticException("Missing rank value when seed is defined is defined for He Normal Initializer");
                    result = CNTKLib.HeNormalInitializer(scale);
                    result = seed.HasValue ? CNTKLib.HeNormalInitializer(scale, rank.Item1, rank.Item2, seed.Value) : CNTKLib.HeNormalInitializer(scale);
                    break;
                case OptInitializers.HeUniform:
                    if (seed.HasValue && rank == null)
                        throw new ArithmeticException("Missing rank value when seed is defined is defined for He Uniform Initializer");
                    result = seed.HasValue ? CNTKLib.HeUniformInitializer(scale, rank.Item1, rank.Item2, seed.Value) : CNTKLib.HeUniformInitializer(scale);
                    break;
                default:
                    break;
            }

            return result;
        }
    }
}
