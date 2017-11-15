using SiaNet.Common;
using System;
using System.Collections.Generic;
using System.Dynamic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SiaNet.Model.Layers
{
    /// <summary>
    /// Turns positive integers (indexes) into dense vectors of fixed size. eg. [[4], [20]] -> [[0.25, 0.1], [0.6, -0.2]]. This layer can only be used as the first layer in a model.
    /// </summary>
    /// <seealso cref="SiaNet.Model.LayerConfig" />
    public class Embedding : LayerConfig
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="Embedding"/> class.
        /// </summary>
        public Embedding()
        {
            base.Name = "Embedding";
            base.Params = new ExpandoObject();
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="Embedding" /> class.
        /// </summary>
        /// <param name="shape">Integer &gt;0. Size of the vocabulary, i.e. maximum integer index + 1.</param>
        /// <param name="embeddingDim">Integer &gt;= 0. Dimension of the dense embedding.</param>
        /// <param name="initializers">Initializer for the embeddings matrix.</param>
        public Embedding(int shape, int embeddingDim, string initializers = OptInitializers.GlorotUniform)
            : this()
        {
            Shape = shape;
            EmbeddingDim = embeddingDim;
        }

        /// <summary>
        /// Integer >0. Size of the vocabulary, i.e. maximum integer index + 1.
        /// </summary>
        /// <value>
        /// The shape.
        /// </value>
        [Newtonsoft.Json.JsonIgnore]
        public double Shape
        {
            get
            {
                return base.Params.Shape;
            }

            set
            {
                base.Params.Shape = value;
            }
        }

        /// <summary>
        /// Integer >= 0. Dimension of the dense embedding.
        /// </summary>
        /// <value>
        /// The embedding dim.
        /// </value>
        [Newtonsoft.Json.JsonIgnore]
        public double EmbeddingDim
        {
            get
            {
                return base.Params.EmbeddingDim;
            }

            set
            {
                base.Params.EmbeddingDim = value;
            }
        }

        /// <summary>
        /// Initializer for the embeddings matrix
        /// </summary>
        /// <value>
        /// The initializers.
        /// </value>
        [Newtonsoft.Json.JsonIgnore]
        public double Initializers
        {
            get
            {
                return base.Params.Initializers;
            }

            set
            {
                base.Params.Initializers = value;
            }
        }
    }
}
