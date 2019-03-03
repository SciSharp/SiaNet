using mx_float = System.Single;

// ReSharper disable once CheckNamespace
namespace SiaNet.Backend.MxNetLib
{

    public sealed partial class Operators
    {

        #region Methods

        public static Symbol FullyConnected(string symbolName,
                                            Symbol data,
                                            Symbol weight,
                                            Symbol bias,
                                            int numHidden,
                                            bool noBias = false,
                                            bool flatten = true)
        {
            return new Operator("FullyConnected").SetParam("num_hidden", numHidden)
                                                 .SetParam("no_bias", noBias)
                                                 .SetParam("flatten", flatten)
                                                 .SetInput("data", data)
                                                 .SetInput("weight", weight)
                                                 .SetInput("bias", bias)
                                                 .CreateSymbol(symbolName);
        }

        public static Symbol FullyConnected(Symbol data,
                                            Symbol weight,
                                            Symbol bias,
                                            int numHidden,
                                            bool no_bias = false,
                                            bool flatten = true)
        {
            return new Operator("FullyConnected").SetParam("num_hidden", numHidden)
                                                 .SetParam("no_bias", no_bias)
                                                 .SetParam("flatten", flatten)
                                                 .SetInput("data", data)
                                                 .SetInput("weight", weight)
                                                 .SetInput("bias", bias)
                                                 .CreateSymbol();
        }

        #endregion

    }

}
