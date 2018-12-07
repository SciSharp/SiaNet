// ReSharper disable once CheckNamespace
namespace SiaNet.Backend
{

    public sealed partial class Operators
    {

        #region Methods

        public static Symbol SliceChannel(string symbolName,
                                          Symbol data,
                                          int numOutputs,
                                          int axis = 1,
                                          bool squeezeAxis = false)
        {
            return new Operator("SliceChannel").SetParam("num_outputs", numOutputs)
                                               .SetParam("axis", axis)
                                               .SetParam("squeeze_axis", squeezeAxis)
                                               .SetInput("data", data)
                                               .CreateSymbol(symbolName);
        }

        public static Symbol SliceChannel(Symbol data,
                                          int numOutputs,
                                          int axis = 1,
                                          bool squeezeAxis = false)
        {
            return new Operator("SliceChannel").SetParam("num_outputs", numOutputs)
                                               .SetParam("axis", axis)
                                               .SetParam("squeeze_axis", squeezeAxis)
                                               .SetInput("data", data)
                                               .CreateSymbol();
        }

        #endregion

    }

}
