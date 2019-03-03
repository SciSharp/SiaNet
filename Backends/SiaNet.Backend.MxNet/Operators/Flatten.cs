// ReSharper disable once CheckNamespace
namespace SiaNet.Backend.MxNetLib
{

    public sealed partial class Operators
    {

        #region Methods

        public static Symbol Flatten(string symbolName, Symbol data)
        {
            return new Operator("Flatten").SetInput("data", data).CreateSymbol(symbolName);
        }

        public static Symbol Flatten(Symbol data)
        {
            return new Operator("Flatten").SetInput("data", data).CreateSymbol();
        }

        #endregion

    }

}
