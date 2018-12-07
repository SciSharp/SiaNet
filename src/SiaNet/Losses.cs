using SiaNet.Backend;
using System;
using System.Collections.Generic;
using System.Text;

namespace SiaNet
{
    public class Losses
    {
        public static Symbol Get(LossType lossType, Symbol preds, Symbol labels)
        {
            switch (lossType)
            {
                case LossType.MeanSquaredError:
                    return MeanSquaredError(preds, labels);
                case LossType.MeanAbsoluteError:
                    return MeanAbsoluteError(preds, labels);
                case LossType.MeanAbsolutePercentageError:
                    return MeanAbsolutePercentageError(preds, labels);
                case LossType.MeanAbsoluteLogError:
                    return MeanAbsoluteLogError(preds, labels);
                case LossType.SquaredHinge:
                    return SquaredHinge(preds, labels);
                case LossType.Hinge:
                    return Hinge(preds, labels);
                case LossType.BinaryCrossEntropy:
                    return BinaryCrossEntropy(preds, labels);
                case LossType.CategorialCrossEntropy:
                    return CategorialCrossEntropy(preds, labels);
                case LossType.CTC:
                    return CTC(preds, labels);
                case LossType.KullbackLeiblerDivergence:
                    return KullbackLeiblerDivergence(preds, labels);
                case LossType.Poisson:
                    return Poisson(preds, labels);
                default:
                    return null;
            }
        }

        private static Symbol MeanSquaredError(Symbol preds, Symbol labels)
        {
            return new Operator("LinearRegressionOutput").SetInput("data", preds).SetInput("label", labels).CreateSymbol();
        }

        private static Symbol MeanAbsoluteError(Symbol preds, Symbol labels)
        {
            return new Operator("MAERegressionOutput").SetInput("data", preds).SetInput("label", labels).CreateSymbol("MeanAbsoluteError");
        }

        private static Symbol MeanAbsolutePercentageError(Symbol preds, Symbol labels)
        {
            Symbol loss = OperatorSupply.Mean(OperatorSupply.Abs(labels - preds) / OperatorSupply.Clip(OperatorSupply.Abs(labels), float.Epsilon, 0));
            return new Operator("MakeLoss").SetInput("data", loss).CreateSymbol("MeanAbsolutePercentageError");
        }

        private static Symbol MeanAbsoluteLogError(Symbol preds, Symbol labels)
        {
            Symbol first_log = OperatorSupply.Log(OperatorSupply.Clip(preds, float.Epsilon, 0) + 1);
            Symbol second_log = OperatorSupply.Log(OperatorSupply.Clip(labels, float.Epsilon, 0) + 1);
            Symbol loss = OperatorSupply.Mean(OperatorSupply.Square(first_log - second_log));
            return new Operator("MakeLoss").SetInput("data", loss).CreateSymbol("MeanAbsoluteLogError");
        }

        private static Symbol SquaredHinge(Symbol preds, Symbol labels)
        {
            Symbol loss = OperatorSupply.Mean(OperatorSupply.Square(OperatorSupply.MaximumScalar(1 - (labels * preds), 0)));
            return new Operator("MakeLoss").SetInput("data", loss).CreateSymbol("SquaredHinge");
        }

        private static Symbol Hinge(Symbol preds, Symbol labels)
        {
            Symbol loss = OperatorSupply.Mean(OperatorSupply.MaximumScalar(1 - (labels * preds), 0));
            return new Operator("MakeLoss").SetInput("data", loss).CreateSymbol("Hinge");
        }

        private static Symbol BinaryCrossEntropy(Symbol preds, Symbol labels)
        {
            return new Operator("LogisticRegressionOutput").SetInput("data", preds).SetInput("label", labels).CreateSymbol("BinaryCrossEntropy");
        }

        private static Symbol CategorialCrossEntropy(Symbol preds, Symbol labels)
        {
            return new Operator("SoftmaxOutput").SetInput("data", preds).SetInput("label", labels).CreateSymbol();
        }

        private static Symbol CTC(Symbol preds, Symbol labels)
        {
            return new Operator("ctc_loss").SetInput("data", preds).SetInput("label", labels).CreateSymbol("CTC");
        }

        private static Symbol KullbackLeiblerDivergence(Symbol preds, Symbol labels)
        {
            Symbol y_true = OperatorSupply.Clip(labels, float.Epsilon, 1);
            Symbol y_pred = OperatorSupply.Clip(preds, float.Epsilon, 1);
            Symbol loss = OperatorSupply.Sum(y_true * OperatorSupply.Log(y_true / y_pred));
            return new Operator("MakeLoss").SetInput("data", loss).CreateSymbol("KullbackLeiblerDivergence");
        }

        private static Symbol Poisson(Symbol preds, Symbol labels)
        {
            Symbol loss = OperatorSupply.Mean(preds - labels * OperatorSupply.Log(preds + float.Epsilon));
            return new Operator("MakeLoss").SetInput("data", loss).CreateSymbol("Poisson");
        }
    }
}
