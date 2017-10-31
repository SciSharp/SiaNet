using CNTK;
using SiaNet.Common;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SiaNet.NN
{
    public class Basic
    {
        public static Function Dense(int shape, int dim, string activation = OptActivations.None, bool useBias = false, string weightInitializer = OptInitializers.Xavier, string biasInitializer = OptInitializers.Zeros)
        {
            var input = CNTKLib.InputVariable(new int[] { shape }, DataType.Float);
            Function fullyConnected = FullyConnected(input, dim, useBias, weightInitializer, biasInitializer);
            return Activation(fullyConnected, activation);
        }

        public static Function Dense(Variable layer, int dim, string activation = OptActivations.None, bool useBias = false, string weightInitializer = OptInitializers.Xavier, string biasInitializer = OptInitializers.Zeros)
        {
            if (layer.Shape.Rank != 1)
            {
                int newDim = layer.Shape.Dimensions.Aggregate((d1, d2) => d1 * d2);
                layer = CNTKLib.Reshape(layer, new int[] { newDim });
            }

            Function fullyConnected = FullyConnected(layer, dim, useBias, weightInitializer, biasInitializer);
            return Activation(fullyConnected, activation);
        }

        public static Function Activation(Variable layer, string activation = OptActivations.ReLU)
        {
            switch (activation.ToLower())
            {
                default:
                case OptActivations.None:
                    return layer;
                case OptActivations.ReLU:
                    return CNTKLib.ReLU(layer);
                case OptActivations.LeakyReLU:
                    return CNTKLib.LeakyReLU(layer);
                case OptActivations.Sigmoid:
                    return CNTKLib.Sigmoid(layer);
                case OptActivations.Tanh:
                    return CNTKLib.Tanh(layer);
                case OptActivations.Softmax:
                    return CNTKLib.Softmax(layer);
                case OptActivations.Softplus:
                    return CNTKLib.Softplus(layer);
                case OptActivations.ELU:
                    return CNTKLib.ELU(layer);
            }
        }

        public static Function Dropout(Variable layer, double rate)
        {
            return CNTKLib.Dropout(layer, rate);
        }

        public static Function BatchNorm(Variable layer, float epsilon=0.001f, string betaInitializer = OptInitializers.Zeros, string gammaInitializers = OptInitializers.Ones,
                                        string runningMeanInitializer = OptInitializers.Zeros, string runningStdInvInitializer = OptInitializers.Ones, bool spatial=true, 
                                        float normalizationTimeConstant=4096f, float blendTimeConst=0.0f)
        {
            var biasParams = new Parameter(new int[] { NDShape.InferredDimension }, DataType.Float, Initializers.Get(betaInitializer), GlobalParameters.Device, "");
            var scaleParams = new Parameter(new int[] { NDShape.InferredDimension }, DataType.Float, Initializers.Get(gammaInitializers), GlobalParameters.Device, "");
            var runningMean = new Parameter(new int[] { NDShape.InferredDimension }, DataType.Float, Initializers.Get(runningMeanInitializer), GlobalParameters.Device, "");
            var runningInvStd = new Constant(new int[] { NDShape.InferredDimension }, 0.0f, GlobalParameters.Device);
            var runningCount = Constant.Scalar(0.0f, GlobalParameters.Device);
            bool useCudnn = false;
            if(GlobalParameters.Device.Type == DeviceKind.GPU)
            {
                useCudnn = true;
            }

            return CNTKLib.BatchNormalization(layer, scaleParams, biasParams, runningMean, runningInvStd, runningCount, spatial, normalizationTimeConstant, blendTimeConst, epsilon, useCudnn);
        }

        public static Function BatchNorm(int shape, float epsilon = 0.001f, string betaInitializer = OptInitializers.Zeros, string gammaInitializers = OptInitializers.Ones,
                                       string runningMeanInitializer = OptInitializers.Zeros, string runningStdInvInitializer = OptInitializers.Ones, bool spatial = true,
                                       float normalizationTimeConstant = 4096f, float blendTimeConst = 0.0f)
        {
            var input = CNTKLib.InputVariable(new int[] { shape }, DataType.Float);
            var biasParams = new Parameter(new int[] { NDShape.InferredDimension }, DataType.Float, Initializers.Get(betaInitializer), GlobalParameters.Device, "");
            var scaleParams = new Parameter(new int[] { NDShape.InferredDimension }, DataType.Float, Initializers.Get(gammaInitializers), GlobalParameters.Device, "");
            var runningMean = new Parameter(new int[] { NDShape.InferredDimension }, DataType.Float, Initializers.Get(runningMeanInitializer), GlobalParameters.Device, "");
            var runningInvStd = new Constant(new int[] { NDShape.InferredDimension }, 0.0f, GlobalParameters.Device);
            var runningCount = Constant.Scalar(0.0f, GlobalParameters.Device);
            bool useCudnn = false;
            if (GlobalParameters.Device.Type == DeviceKind.GPU)
            {
                useCudnn = true;
            }

            return CNTKLib.BatchNormalization(input, scaleParams, biasParams, runningMean, runningInvStd, runningCount, spatial, normalizationTimeConstant, blendTimeConst, epsilon, useCudnn);
        }

        private static Function FullyConnected(Variable input, int outputDim, bool useBias, string weightInitializer, string biasInitializer)
        {
            int inputDim = input.Shape[0];
            
            int[] s = { outputDim, inputDim };
            var weights = new Parameter(s, DataType.Float, Initializers.Get(weightInitializer), GlobalParameters.Device);

            Parameter bias = null;
            if (useBias)
            {
                int[] s2 = { outputDim };
                bias = new Parameter(s2, DataType.Float, Initializers.Get(biasInitializer), GlobalParameters.Device);
            }
            else
            {
                int[] s2 = { outputDim };
                bias = new Parameter(s2, DataType.Float, 0.0f, GlobalParameters.Device);
            }
            
            return CNTKLib.Plus(bias, CNTKLib.Times(weights, input));
        }
    }
}
