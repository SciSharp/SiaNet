using uint32_t = System.UInt32;

// ReSharper disable once CheckNamespace
namespace SiaNet.Backend.MxNetLib
{

    public sealed partial class Operators
    {

        #region Methods

        public static Symbol SwapAxis(string symbolName,
                                      Symbol data,
                                      uint32_t dim1 = 0,
                                      uint32_t dim2 = 0)
        {
            return new Operator("SwapAxis").SetParam("dim1", dim1)
                                           .SetParam("dim2", dim2)
                                           .SetInput("data", data)
                                           .CreateSymbol(symbolName);
        }

        public static Symbol SwapAxis(Symbol data,
                                      uint32_t dim1 = 0,
                                      uint32_t dim2 = 0)
        {
            return new Operator("SwapAxis").SetParam("dim1", dim1)
                                           .SetParam("dim2", dim2)
                                           .SetInput("data", data)
                                           .CreateSymbol();
        }

        #endregion

    }

}
