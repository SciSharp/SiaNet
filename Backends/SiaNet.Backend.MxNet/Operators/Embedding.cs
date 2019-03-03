using mx_float = System.Single;

// ReSharper disable once CheckNamespace
namespace SiaNet.Backend.MxNetLib
{

    public sealed partial class Operators
    {

        #region Fields

        private static readonly string[] EmbeddingDtypeValues =
        {
            "float16",
            "float32",
            "float64",
            "int32",
            "uint8"
        };

        #endregion

        #region Methods

        public static Symbol Embedding(string symbolName,
                                       Symbol data,
                                       Symbol weight,
                                       int inputDim,
                                       int outputDim,
                                       EmbeddingDtype dtype = EmbeddingDtype.Float32)
        {
            return new Operator("Embedding").SetParam("input_dim", inputDim)
                                            .SetParam("output_dim", outputDim)
                                            .SetParam("dtype", EmbeddingDtypeValues[(int)dtype])
                                            .SetInput("data", data)
                                            .SetInput("weight", weight)
                                            .CreateSymbol(symbolName);
        }

        public static Symbol Embedding(Symbol data,
                                       Symbol weight,
                                       int inputDim,
                                       int outputDim,
                                       EmbeddingDtype dtype = EmbeddingDtype.Float32)
        {
            return new Operator("Embedding").SetParam("input_dim", inputDim)
                                            .SetParam("output_dim", outputDim)
                                            .SetParam("dtype", EmbeddingDtypeValues[(int)dtype])
                                            .SetInput("data", data)
                                            .SetInput("weight", weight)
                                            .CreateSymbol();
        }

        #endregion

    }

}
