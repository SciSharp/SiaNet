using Newtonsoft.Json;
using SiaNet.Model.Initializers;
using SiaNet.NN;

namespace SiaNet.Model.Layers
{
    /// <summary>
    ///     Turns positive integers (indexes) into dense vectors of fixed size. eg. [[4], [20]] -> [[0.25, 0.1], [0.6, -0.2]].
    ///     This layer can only be used as the first layer in a model.
    /// </summary>
    /// <seealso cref="OptimizableLayerBase" />
    public class Embedding : OptimizableLayerBase
    {
        /// <summary>
        ///     Initializes a new instance of the <see cref="Embedding" /> class.
        /// </summary>
        /// <param name="shape">Integer &gt;0. Size of the vocabulary, i.e. maximum integer index + 1.</param>
        /// <param name="embeddingDim">Integer &gt;= 0. Dimension of the dense embedding.</param>
        /// <param name="initializers">Initializer for the embeddings matrix.</param>
        public Embedding(int embeddingDim, InitializerBase initializers = null)
        {
            EmbeddingDim = embeddingDim;
            Initializers = initializers ?? new GlorotUniform();
        }

        /// <summary>
        ///     Integer >= 0. Dimension of the dense embedding.
        /// </summary>
        /// <value>
        ///     The embedding dim.
        /// </value>
        [JsonIgnore]
        public int EmbeddingDim
        {
            get => GetParam<int>("EmbeddingDim");

            set => SetParam("EmbeddingDim", value);
        }

        /// <summary>
        ///     Initializer for the embeddings matrix
        /// </summary>
        /// <value>
        ///     The initializers.
        /// </value>
        [JsonIgnore]
        public InitializerBase Initializers
        {
            get => GetParam<InitializerBase>("Initializers");

            set => SetParam("Initializers", value);
        }

        /// <inheritdoc />
        internal override Function ToFunction(Variable inputFunction)
        {
            //if (inputFunction.Shape.Rank != 1)
            //{
            //    throw new ArgumentException("Variable has an invalid shape.", nameof(inputFunction));
            //}

            return Basic.Embedding(inputFunction, EmbeddingDim, Initializers);
        }
    }
}