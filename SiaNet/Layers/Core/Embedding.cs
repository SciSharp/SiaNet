namespace SiaNet.Layers
{
    using SiaNet.Constraints;
    using SiaNet.Engine;
    using SiaNet.Initializers;
    using SiaNet.Regularizers;

    /// <summary>
    /// Turns positive integers (indexes) into dense vectors of fixed size. eg. [[4], [20]] -> [[0.25, 0.1], [0.6, -0.2]]
    /// <para>
    /// This layer can only be used as the first layer in a model.
    /// </para>
    /// </summary>
    /// <seealso cref="SiaNet.Layers.BaseLayer" />
    public class Embedding : BaseLayer
    {
        /// <summary>
        /// int > 0, Size of the vocabulary, i.e. maximum integer index + 1.
        /// </summary>
        /// <value>
        /// The input dim.
        /// </value>
        public int InputDim { get; set; }

        /// <summary>
        /// int >= 0, Dimension of the dense embedding.
        /// </summary>
        /// <value>
        /// The output dim.
        /// </value>
        public int OutputDim { get; set; }

        /// <summary>
        /// Initializer for the embedding weight matrix.
        /// </summary>
        /// <value>
        /// The embeddings initializer.
        /// </value>
        public BaseInitializer EmbeddingsInitializer { get; set; }

        /// <summary>
        /// Constraint function for the embedding weight matrix
        /// </summary>
        /// <value>
        /// The embeddings constraint.
        /// </value>
        public BaseConstraint EmbeddingsConstraint { get; set; }

        /// <summary>
        /// Regularizer function for the embedding weight matrix.
        /// </summary>
        /// <value>
        /// The embeddings regularizer.
        /// </value>
        public BaseRegularizer EmbeddingsRegularizer { get; set; }

        /// <summary>
        /// Initializes a new instance of the <see cref="Embedding"/> class.
        /// </summary>
        /// <param name="inputDim">Size of the vocabulary, i.e. maximum integer index + 1.</param>
        /// <param name="outputDim">int >= 0, Dimension of the dense embedding.</param>
        /// <param name="embeddingsInitializer">Initializer for the embedding weight matrix.</param>
        /// <param name="embeddingsRegularizer">Regularizer function for the embedding weight matrix.</param>
        /// <param name="embeddingsConstraint">Constraint function for the embedding weight matrix.</param>
        public Embedding(int inputDim, int outputDim, BaseInitializer embeddingsInitializer = null, BaseRegularizer embeddingsRegularizer = null, BaseConstraint embeddingsConstraint = null)
            : base("embedding")
        {
            InputDim = inputDim;
            OutputDim = outputDim;
            EmbeddingsInitializer = embeddingsInitializer ?? new RandomUniform();
            EmbeddingsConstraint = embeddingsConstraint;
            EmbeddingsRegularizer = embeddingsRegularizer;
        }

        /// <summary>
        /// Forwards the inputs and compute the output
        /// </summary>
        /// <param name="x">The input tensor for this layer.</param>
        public override void Forward(Tensor x)
        {
            base.Forward(x);
        }

        /// <summary>
        /// Calculate the gradient of this layer function
        /// </summary>
        /// <param name="outputgrad">The calculated output grad from previous layer.</param>
        public override void Backward(Tensor outputgrad)
        {
            
        }
    }
}
