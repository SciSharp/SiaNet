using System;
using System.Collections.Generic;
using System.Text;
using TensorSharp.Expression;

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

        public override TVar Call(TVar preds, TVar labels)
        {
            //var indices = labels.View(labels.Evaluate().Sizes);

            //var loss = preds.Gather(1, indices)
            //    .SumAll()
            //     * (-1.0f / labels.Evaluate().Sizes[0]);
            preds.Print();
            labels.Print();
            var loss = -1f * labels.CMul(preds.Log()).SumAll();

            return loss;
        }

        public override TVar CalcGrad(TVar preds, TVar labels)
        {
            var m = labels.Evaluate().Sizes[0];
            var grad = preds.Softmax();
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
