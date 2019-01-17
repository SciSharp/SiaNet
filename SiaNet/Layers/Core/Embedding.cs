using System;
using System.Collections.Generic;
using System.Text;
using SiaNet.Constraints;
using SiaNet.Initializers;
using SiaNet.Layers.Activations;
using SiaNet.Regularizers;
using TensorSharp;
using TensorSharp.Expression;

namespace SiaNet.Layers
{
    public class Embedding : BaseLayer
    {
        public int InputDim { get; set; }

        public int OutputDim { get; set; }

        public BaseInitializer EmbeddingsInitializer { get; set; }

        public BaseConstraint EmbeddingsConstraint { get; set; }

        public BaseRegularizer EmbeddingsRegularizer { get; set; }

        public Embedding(int inputDim, int outputDim, BaseInitializer embeddingsInitializer = null, BaseRegularizer embeddingsRegularizer = null, BaseConstraint embeddingsConstraint = null)
            : base("embedding")
        {
            InputDim = inputDim;
            OutputDim = outputDim;
            EmbeddingsInitializer = embeddingsInitializer ?? new RandomUniform();
            EmbeddingsConstraint = embeddingsConstraint;
            EmbeddingsRegularizer = embeddingsRegularizer;
        }

        public override void Forward(Parameter x)
        {
            
        }

        public override void Backward(Tensor outputgrad)
        {
            
        }
    }
}
