using mx_float = System.Single;
using uint32_t = System.UInt32;

// ReSharper disable once CheckNamespace
namespace SiaNet.Backend.MxNetLib
{

    public sealed partial class Operators
    {

        #region Fields

        private static readonly string[] RNNModeValues =
        {
            "gru",
            "lstm",
            "rnn_relu",
            "rnn_tanh"
        };

        #endregion

        #region Methods

        public static Symbol RNN(string symbolName,
                                 Symbol data,
                                 Symbol parameters,
                                 Symbol state,
                                 Symbol stateCell,
                                 uint32_t stateSize,
                                 uint32_t numLayers,
                                 RNNMode mode,
                                 bool bidirectional = false,
                                 mx_float p = 0,
                                 bool stateOutputs = false)
        {
            return new Operator("RNN").SetParam("state_size", stateSize)
                                      .SetParam("num_layers", numLayers)
                                      .SetParam("mode", RNNModeValues[(int)mode])
                                      .SetParam("bidirectional", bidirectional)
                                      .SetParam("p", p)
                                      .SetParam("state_outputs", stateOutputs)
                                      .SetInput("data", data)
                                      .SetInput("parameters", parameters)
                                      .SetInput("state", state)
                                      .SetInput("state_cell", stateCell)
                                      .CreateSymbol(symbolName);
        }

        public static Symbol RNN(Symbol data,
                                 Symbol parameters,
                                 Symbol state,
                                 Symbol stateCell,
                                 uint32_t stateSize,
                                 uint32_t numLayers,
                                 RNNMode mode,
                                 bool bidirectional = false,
                                 mx_float p = 0,
                                 bool stateOutputs = false)
        {
            return new Operator("RNN").SetParam("state_size", stateSize)
                                      .SetParam("num_layers", numLayers)
                                      .SetParam("mode", RNNModeValues[(int)mode])
                                      .SetParam("bidirectional", bidirectional)
                                      .SetParam("p", p)
                                      .SetParam("state_outputs", stateOutputs)
                                      .SetInput("data", data)
                                      .SetInput("parameters", parameters)
                                      .SetInput("state", state)
                                      .SetInput("state_cell", stateCell)
                                      .CreateSymbol();
        }

        #endregion

    }

}
