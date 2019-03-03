// ReSharper disable once CheckNamespace
namespace SiaNet.Backend.MxNetLib
{

    public sealed partial class Operators
    {

        #region Methods

        public static Symbol transpose(string symbolName, Symbol data)
        {
            return transpose(symbolName, data, new Shape());
        }

        public static Symbol transpose(string symbolName, Symbol data, Shape axes)
        {
            return new Operator("transpose").SetParam("axes", axes)
                                            .SetInput("data", data)
                                            .CreateSymbol(symbolName);
        }

        public static Symbol transpose(Symbol data)
        {
            return transpose(data, new Shape());
        }

        public static Symbol transpose(Symbol data, Shape axes)
        {
            return new Operator("transpose").SetParam("axes", axes)
                                            .SetInput("data", data)
                                            .CreateSymbol();
        }

        #endregion

    }

}
