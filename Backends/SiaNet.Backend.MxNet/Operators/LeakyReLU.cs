using mx_float = System.Single;

// ReSharper disable once CheckNamespace
namespace SiaNet.Backend.MxNetLib
{

    public sealed partial class Operators
    {

        #region Fields

        private static readonly string[] LeakyReLUActTypeValues =
        {
            "elu",
            "leaky",
            "prelu",
            "rrelu"
        };

        #endregion

        #region Methods

        public static Symbol LeakyReLU(string symbolName,
                                       Symbol data,
                                       LeakyReLUActType actType = LeakyReLUActType.Leaky,
                                       mx_float slope = 0.25f,
                                       mx_float lowerBound = 0.125f,
                                       mx_float upperBound = 0.334f)
        {
            return new Operator("LeakyReLU").SetParam("act_type", LeakyReLUActTypeValues[(int)actType])
                                            .SetParam("slope", slope)
                                            .SetParam("lower_bound", lowerBound)
                                            .SetParam("upper_bound", upperBound)
                                            .SetInput("data", data)
                                            .CreateSymbol(symbolName);
        }

        public static Symbol LeakyReLU(Symbol data,
                                       LeakyReLUActType actType = LeakyReLUActType.Leaky,
                                       mx_float slope = 0.25f,
                                       mx_float lowerBound = 0.125f,
                                       mx_float upperBound = 0.334f)
        {
            return new Operator("LeakyReLU").SetParam("act_type", LeakyReLUActTypeValues[(int)actType])
                                            .SetParam("slope", slope)
                                            .SetParam("lower_bound", lowerBound)
                                            .SetParam("upper_bound", upperBound)
                                            .SetInput("data", data)
                                            .CreateSymbol();
        }

        #endregion

    }

}
