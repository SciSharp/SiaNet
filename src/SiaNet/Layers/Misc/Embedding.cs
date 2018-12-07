using System;
using System.Collections.Generic;
using System.Text;
using SiaDNN.Constraints;
using SiaDNN.Initializers;
using SiaNet.Backend;
using SiaNet.Regularizers;

namespace SiaNet.Layers.Misc
{
    public class Embedding : BaseLayer, ILayer
    {
        public int InputDim { get; set; }

        public int OutputDim { get; set; }

        public BaseInitializer EmbeddingsInitializer { get; set; }

        public BaseConstraint EmbeddingsConstraint { get; set; }

        public BaseRegularizer EmbeddingsRegularizer { get; set; }

        public Embedding(int inputDim, int outputDim, BaseInitializer embeddingsInitializer=null, BaseRegularizer embeddingsRegularizer=null,BaseConstraint embeddingsConstraint = null)
            :base("embedding")
        {
            InputDim = inputDim;
            OutputDim = outputDim;
            EmbeddingsInitializer = embeddingsInitializer ?? new RandomUniform();
            EmbeddingsConstraint = embeddingsConstraint;
            EmbeddingsRegularizer = embeddingsRegularizer;
        }

        public Symbol Build(Symbol x)
        {
            var weightName = UUID.GetID(ID + "_w");
            InitParams.Add(weightName, EmbeddingsInitializer);
            ConstraintParams.Add(weightName, EmbeddingsConstraint);
            RegularizerParams.Add(weightName, EmbeddingsRegularizer);
            return Operators.Embedding(ID, x, Symbol.Variable(weightName), InputDim, OutputDim);
        }
    }
}
