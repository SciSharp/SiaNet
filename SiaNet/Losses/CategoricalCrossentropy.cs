using System;
using System.Collections.Generic;
using System.Text;
using TensorSharp;

namespace SiaNet.Losses
{
    public class CategoricalCrossentropy : BaseLoss
    {
        public bool FromLogit { get; set; }

        public CategoricalCrossentropy(bool fromLogit = false)
            : base("categorical_crossentropy")
        {
            FromLogit = fromLogit;
        }

        public override Tensor Call(Tensor preds, Tensor labels)
        {
            var loss = -1f * Sum(labels * Log(preds));

            return loss;
        }

        public override Tensor CalcGrad(Tensor preds, Tensor labels)
        {
            var m = labels.Sizes[0];
            var grad = Softmax(preds);
            grad = preds - 1;
            grad = grad / m;
            return grad;
            //var norm = -1.0f / preds.Evaluate().Sizes[0];
            //var labelShape = labels.Evaluate().Sizes;
            //TVar gradInput = TVar.Fill(0, Global.Device, DType.Float32, labelShape);

            //var indices = labels.View(labelShape);

            //return gradInput.ScatterFill(norm, 1, indices);
        }
    }
}
