using CNTK;
using SiaNet.Common;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SiaNet.NN
{
    public class Recurrent
    {
        public static Function LSTM(int inputDim, uint hiddenSize, uint numLayers, bool bidirectional = false, string weightInitializer = OptInitializers.Xavier)
        {
            return BuildRNN(inputDim, hiddenSize, numLayers, bidirectional, weightInitializer, "lstm");
        }

        public static Function GRU(int inputDim, uint hiddenSize, uint numLayers, bool bidirectional = false, string weightInitializer = OptInitializers.Xavier)
        {
            return BuildRNN(inputDim, hiddenSize, numLayers, bidirectional, weightInitializer, "gru");
        }

        public static Function RNN(int inputDim, uint hiddenSize, uint numLayers, string activation, bool bidirectional = false, string weightInitializer = OptInitializers.Xavier)
        {
            switch (activation)
            {
                case OptActivations.ReLU:
                    return BuildRNN(inputDim, hiddenSize, numLayers, bidirectional, weightInitializer, "rnnReLU");
                case OptActivations.Tanh:
                    return BuildRNN(inputDim, hiddenSize, numLayers, bidirectional, weightInitializer, "rnnTanh");
                default:
                    throw new Exception("Supported activation for RNN is ReLU and Tanh");
            }
        }

        private static Function BuildRNN(int inputDim, uint hiddenSize, uint numLayers, bool bidirectional = false, string weightInitializer = OptInitializers.Xavier, string rnnName = "")
        {
            int[] s = { inputDim };
            var weights = new Parameter(s, DataType.Float, Initializers.Get(weightInitializer), GlobalParameters.Device);

            return CNTKLib.OptimizedRNNStack(Variable.InputVariable(new int[] { inputDim }, DataType.Float), weights, hiddenSize, numLayers, bidirectional, rnnName);
        }
    }
}
