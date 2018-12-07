using mx_float = System.Single;

// ReSharper disable once CheckNamespace
namespace SiaNet.Backend
{

    public sealed partial class Operators
    {

        #region Methods

        public static Symbol BatchNorm(string symbolName,
                                       Symbol data,
                                       Symbol gamma,
                                       Symbol beta,
                                       Symbol movingMean,
                                       Symbol movingVar,
                                       double eps = 0.001,
                                       mx_float momentum = 0.9f,
                                       bool fixGamma = true,
                                       bool useGlobalStats = false,
                                       bool outputMeanVar = false,
                                       int axis = 1,
                                       bool cudnnOff = false)
        {
            return new Operator("BatchNorm").SetParam("eps", eps)
                                            .SetParam("momentum", momentum)
                                            .SetParam("fix_gamma", fixGamma)
                                            .SetParam("use_global_stats", useGlobalStats)
                                            .SetParam("output_mean_var", outputMeanVar)
                                            .SetParam("axis", axis)
                                            .SetParam("cudnn_off", cudnnOff)
                                            .SetInput("data", data)
                                            .SetInput("gamma", gamma)
                                            .SetInput("beta", beta)
                                            .SetInput("moving_mean", movingMean)
                                            .SetInput("moving_var", movingVar)
                                            .CreateSymbol(symbolName);
        }

        public static Symbol BatchNorm(Symbol data,
                                           Symbol gamma,
                                           Symbol beta,
                                           Symbol movingMean,
                                           Symbol movingVar,
                                           double eps = 0.001,
                                           mx_float momentum = 0.9f,
                                           bool fixGamma = true,
                                           bool useGlobalStats = false,
                                           bool outputMeanVar = false,
                                           int axis = 1,
                                           bool cudnnOff = false)
        {
            return new Operator("BatchNorm").SetParam("eps", eps)
                                            .SetParam("momentum", momentum)
                                            .SetParam("fix_gamma", fixGamma)
                                            .SetParam("use_global_stats", useGlobalStats)
                                            .SetParam("output_mean_var", outputMeanVar)
                                            .SetParam("axis", axis)
                                            .SetParam("cudnn_off", cudnnOff)
                                            .SetInput("data", data)
                                            .SetInput("gamma", gamma)
                                            .SetInput("beta", beta)
                                            .SetInput("moving_mean", movingMean)
                                            .SetInput("moving_var", movingVar)
                                            .CreateSymbol();
        }

        #endregion

    }

}
