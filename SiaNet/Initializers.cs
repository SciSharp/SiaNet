using CNTK;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SiaNet
{
    public class Initializers
    {
        public static CNTKDictionary Get(string initializers)
        {
            CNTKDictionary result = null;
            switch (initializers.Trim().ToLower())
            {
                case OptInitializers.Uniform:
                    result = CNTKLib.UniformInitializer(CNTKLib.DefaultParamInitScale, 1);
                    break;
                case OptInitializers.Normal:
                    result = CNTKLib.NormalInitializer(CNTKLib.DefaultParamInitScale, 1);
                    break;
                case OptInitializers.TruncatedNormal:
                    result = CNTKLib.TruncatedNormalInitializer(CNTKLib.DefaultParamInitScale, 1);
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
                    result = CNTKLib.XavierInitializer(CNTKLib.DefaultParamInitScale, CNTKLib.SentinelValueForInferParamInitRank, CNTKLib.SentinelValueForInferParamInitRank, 1);
                    break;
                case OptInitializers.GlorotNormal:
                    result = CNTKLib.GlorotNormalInitializer(CNTKLib.DefaultParamInitScale, CNTKLib.SentinelValueForInferParamInitRank, CNTKLib.SentinelValueForInferParamInitRank, 1);
                    break;
                case OptInitializers.GlorotUniform:
                    result = CNTKLib.GlorotUniformInitializer(CNTKLib.DefaultParamInitScale, CNTKLib.SentinelValueForInferParamInitRank, CNTKLib.SentinelValueForInferParamInitRank, 1);
                    break;
                case OptInitializers.HeNormal:
                    result = CNTKLib.HeNormalInitializer(CNTKLib.DefaultParamInitScale, CNTKLib.SentinelValueForInferParamInitRank, CNTKLib.SentinelValueForInferParamInitRank, 1);
                    break;
                case OptInitializers.HeUniform:
                    result = CNTKLib.HeUniformInitializer(CNTKLib.DefaultParamInitScale, CNTKLib.SentinelValueForInferParamInitRank, CNTKLib.SentinelValueForInferParamInitRank, 1);
                    break;
                default:
                    break;
            }

            return result;
        }
    }
}
