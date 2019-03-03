using mx_float = System.Single;

// ReSharper disable once CheckNamespace
namespace SiaNet.Backend.MxNetLib
{

    public sealed partial class Operators
    {

        #region Fields

        private static readonly string[] DropoutModeValues =
        {
            "always",
            "training"
        };

        #endregion

        #region Methods

        public static Symbol Dropout(string symbolName,
                                     Symbol data,
                                     mx_float p = 0.5f,
                                     DropoutMode mode = DropoutMode.Training)
        {
            return new Operator("Dropout").SetParam("p", p)
                                          .SetParam("mode", DropoutModeValues[(int)mode])
                                          .SetInput("data", data)
                                          .CreateSymbol(symbolName);
        }

        public static Symbol Dropout(Symbol data,
                                     mx_float p = 0.5f,
                                     DropoutMode mode = DropoutMode.Training)
        {
            return new Operator("Dropout").SetParam("p", p)
                                          .SetParam("mode", DropoutModeValues[(int)mode])
                                          .SetInput("data", data)
                                          .CreateSymbol();
        }

        #endregion

    }

}
