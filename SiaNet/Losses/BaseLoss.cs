using System;
using System.Collections.Generic;
using System.Text;
using TensorSharp;

namespace SiaNet.Losses
{
    public abstract class BaseLoss : TOps
    {
        public string Name { get; set; }

        public BaseLoss(string name)
        {
            Name = name;
        }

        public abstract Tensor Call(Tensor preds, Tensor labels);

        public abstract Tensor CalcGrad(Tensor preds, Tensor labels);

        public static BaseLoss Get(LossType lossType)
        {
            BaseLoss loss = null;
            switch (lossType)
            {
                case LossType.MeanSquaredError:
                    loss = new MeanSquaredError();
                    break;
                case LossType.MeanAbsoluteError:
                    loss = new MeanAbsoluteError();
                    break;
                case LossType.MeanAbsolutePercentageError:
                    loss = new MeanAbsolutePercentageError();
                    break;
                case LossType.MeanAbsoluteLogError:
                    loss = new MeanSquaredLogError();
                    break;
                case LossType.SquaredHinge:
                    loss = new SquaredHinge();
                    break;
                case LossType.Hinge:
                    loss = new Hinge();
                    break;
                case LossType.BinaryCrossEntropy:
                    loss = new BinaryCrossentropy();
                    break;
                case LossType.CategorialCrossEntropy:
                    loss = new CategoricalCrossentropy();
                    break;
                case LossType.CTC:
                    break;
                case LossType.KullbackLeiblerDivergence:
                    loss = new KullbackLeiblerDivergence();
                    break;
                case LossType.Logcosh:
                    loss = new LogCosh();
                    break;
                case LossType.Poisson:
                    loss = new Poisson();
                    break;
                case LossType.CosineProximity:
                    loss = new CosineProximity();
                    break;
                default:
                    break;
            }

            return loss;
        }
    }
}
