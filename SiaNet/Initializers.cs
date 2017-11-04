using CNTK;
using SiaNet.Common;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SiaNet
{
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
        public static CNTKDictionary Get(string initializers)
        {
            CNTKDictionary result = null;
            switch (initializers.Trim().ToLower())
            {
                case OptInitializers.Uniform:
                    result = CNTKLib.UniformInitializer(0.01);
                    break;
                case OptInitializers.Normal:
                    result = CNTKLib.NormalInitializer(0.01);
                    break;
                case OptInitializers.TruncatedNormal:
                    result = CNTKLib.TruncatedNormalInitializer();
                    break;
                case OptInitializers.Zeros:
                    result = CNTKLib.ConstantInitializer(0);
                    break;
                case OptInitializers.Ones:
                    result = CNTKLib.ConstantInitializer(1);
                    break;
                case OptInitializers.Constant:
                    result = CNTKLib.ConstantInitializer();
                    break;
                case OptInitializers.Xavier:
                    result = CNTKLib.XavierInitializer();
                    break;
                case OptInitializers.GlorotNormal:
                    result = CNTKLib.GlorotNormalInitializer();
                    break;
                case OptInitializers.GlorotUniform:
                    result = CNTKLib.GlorotUniformInitializer();
                    break;
                case OptInitializers.HeNormal:
                    result = CNTKLib.HeNormalInitializer();
                    break;
                case OptInitializers.HeUniform:
                    result = CNTKLib.HeUniformInitializer();
                    break;
                default:
                    break;
            }

            return result;
        }
    }
}
