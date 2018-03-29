using Newtonsoft.Json;
using SiaNet.Common;
using SiaNet.Model.Initializers;

namespace SiaNet.Model.Layers
{
    /// <summary>
    ///     Turns positive integers (indexes) into dense vectors of fixed size. eg. [[4], [20]] -> [[0.25, 0.1], [0.6, -0.2]].
    ///     This layer can only be used as the first layer in a model.
    /// </summary>
    /// <seealso cref="SiaNet.Model.LayerConfig" />
    public class Embedding : LayerConfig
    {
        /// <summary>
        ///     Initializes a new instance of the <see cref="Embedding" /> class.
        /// </summary>
        /// <param name="shape">Integer &gt;0. Size of the vocabulary, i.e. maximum integer index + 1.</param>
        /// <param name="embeddingDim">Integer &gt;= 0. Dimension of the dense embedding.</param>
        /// <param name="initializers">Initializer for the embeddings matrix.</param>
        public Embedding(int shape, int embeddingDim, string initializers = OptInitializers.GlorotUniform)
            : this()
        {
            Shape = shape;
            EmbeddingDim = embeddingDim;
            Initializers = new Initializer(initializers);
        }

        /// <summary>
        ///     Initializes a new instance of the <see cref="Embedding" /> class.
        /// </summary>
        /// <param name="shape">Integer &gt;0. Size of the vocabulary, i.e. maximum integer index + 1.</param>
        /// <param name="embeddingDim">Integer &gt;= 0. Dimension of the dense embedding.</param>
        /// <param name="initializers">Initializer for the embeddings matrix.</param>
        public Embedding(int shape, int embeddingDim, Initializer initializers = null)
            : this()
        {
            Shape = shape;
            EmbeddingDim = embeddingDim;
            Initializers = initializers;
        }

        /// <summary>
        ///     Initializes a new instance of the <see cref="Embedding" /> class.
        /// </summary>
        internal Embedding()
        {
            Name = "Embedding";
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
        public Initializer Initializers
        {
            get => GetParam<Initializer>("Initializers");

            set => SetParam("Initializers", value);
        }

        /// <summary>
        ///     Integer >0. Size of the vocabulary, i.e. maximum integer index + 1.
        /// </summary>
        /// <value>
        ///     The shape.
        /// </value>
        [JsonIgnore]
        public int Shape
        {
            get => GetParam<int>("Shape");

            set => SetParam("Shape", value);
        }
    }
}