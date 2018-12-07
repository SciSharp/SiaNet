using System.Collections.Generic;

// ReSharper disable once CheckNamespace
namespace SiaNet.Backend
{

    public sealed partial class Operators
    {

        #region Methods

        public static Symbol Concat(string symbolName,
                                    IList<Symbol> data,
                                    int numArgs,
                                    int dim = 1)
        {
            return new Operator("Concat").SetParam("num_args", numArgs)
                                         .SetParam("dim", dim)
                                         .Set(data)
                                         .CreateSymbol(symbolName);
        }

        public static Symbol Concat(IList<Symbol> data,
                                    int numArgs,
                                    int dim = 1)
        {
            return new Operator("Concat").SetParam("num_args", numArgs)
                                         .SetParam("dim", dim)
                                         .Set(data)
                                         .CreateSymbol();
        }

        #endregion

    }

}
