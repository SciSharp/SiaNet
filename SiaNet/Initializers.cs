namespace SiaNet
{
    using CNTK;
    using SiaNet.Common;

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
        /// <returns>Initializer instance as CNTKDictionary.</returns>
        public static CNTKDictionary Get(string initializers, double scale = 0.1)
        {
            CNTKDictionary result = null;
            switch (initializers.Trim().ToLower())
            {
                case OptInitializers.Uniform:
                    result = CNTKLib.UniformInitializer(scale);
                    break;
                case OptInitializers.Normal:
                    result = CNTKLib.NormalInitializer(scale);
                    break;
                case OptInitializers.TruncatedNormal:
                    result = CNTKLib.TruncatedNormalInitializer(scale);
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
                    result = CNTKLib.XavierInitializer(scale);
                    break;
                case OptInitializers.GlorotNormal:
                    result = CNTKLib.GlorotNormalInitializer(scale);
                    break;
                case OptInitializers.GlorotUniform:
                    result = CNTKLib.GlorotUniformInitializer(scale);
                    break;
                case OptInitializers.HeNormal:
                    result = CNTKLib.HeNormalInitializer(scale);
                    break;
                case OptInitializers.HeUniform:
                    result = CNTKLib.HeUniformInitializer(scale);
                    break;
                default:
                    break;
            }

            return result;
        }
    }
}
