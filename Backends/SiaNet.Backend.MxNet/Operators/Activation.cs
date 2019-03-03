using mx_float = System.Single;

// ReSharper disable once CheckNamespace
namespace SiaNet.Backend.MxNetLib
{

    public sealed partial class Operators
    {

        #region Fields

        private static string[] ActivationActTypeValues =
        {
            "relu",
            "sigmoid",
            "softrelu",
            "tanh"
        };

        #endregion

        #region Methods

        public static Symbol Activation(string symbolName,
                                        Symbol data,
                                        ActivationActType actType)
        {
            return new Operator("Activation").SetParam("act_type", ActivationActTypeValues[(int)actType])
                                             .SetInput("data", data)
                                             .CreateSymbol(symbolName);
        }

        public static Symbol Activation(Symbol data, ActivationActType actType)
        {
            return new Operator("Activation").SetParam("act_type", ActivationActTypeValues[(int)actType])
                                             .SetInput("data", data)
                                             .CreateSymbol();
        }

        #endregion

    }

}
