using uint32_t = System.UInt32;

// ReSharper disable once CheckNamespace
namespace SiaNet.Backend.MxNetLib
{

    public sealed partial class Operators
    {

        #region Methods

        public static Symbol Reshape(string symbolName,
                                     Symbol data)
        {
            return Reshape(symbolName, data, new Shape());
        }

        public static Symbol Reshape(string symbolName,
                                     Symbol data,
                                     Shape shape)
        {
            return Reshape(symbolName, data, shape);
        }

        public static Symbol Reshape(string symbolName,
                                     Symbol data,
                                     Shape shape,
                                     bool reverse = false)
        {
            return new Operator("reshape").SetParam("shape", shape)
                                          .SetParam("reverse", reverse)
                                          .SetInput("data", data)
                                          .CreateSymbol(symbolName);
        }

        public static Symbol Reshape(Symbol data)
        {
            return Reshape(data, new Shape());
        }

        public static Symbol Reshape(Symbol data,
                                     Shape shape)
        {
            return Reshape(data, shape);
        }

        public static Symbol Reshape(Symbol data,
                                     Shape shape,
                                     bool reverse = false)
        {
            return new Operator("reshape").SetParam("shape", shape)
                                          .SetParam("reverse", reverse)
                                          .SetInput("data", data)
                                          .CreateSymbol();
        }

        #endregion

    }

}
