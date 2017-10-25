using CNTK;
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
            int[] s = { inputDim };
            var weights = new Parameter(s, DataType.Float, Initializers.Get(weightInitializer), GlobalParameters.Device);

            return CNTKLib.OptimizedRNNStack(null, weights, hiddenSize, numLayers, bidirectional, "lstm");
        }
    }
}
