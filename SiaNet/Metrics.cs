using CNTK;
using SiaNet.Common;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SiaNet
{
    public class Metrics
    {
        public static Function Get(string metric, Variable labels, Variable predictions)
        {
            switch (metric.Trim().ToLower())
            {
                case OptMetrics.Accuracy:
                    return Accuracy(labels, predictions);
                case OptMetrics.MAE:
                    return MAE(labels, predictions);
                case OptMetrics.MAPE:
                    return MAPE(labels, predictions);
                case OptMetrics.MSE:
                    return MSE(labels, predictions);
                case OptMetrics.MSLE:
                    return MSLE(labels, predictions);
                case OptMetrics.TopKAccuracy:
                    return TopKAccuracy(labels, predictions);
                default:
                    throw new NotImplementedException(string.Format("{0} is not implemented", metric));
            }
        }

        private static Function Accuracy(Variable labels, Variable predictions)
        {
            return CNTKLib.ClassificationError(predictions, labels);
        }

        private static Function TopKAccuracy(Variable labels, Variable predictions, uint k=5)
        {
            return CNTKLib.ClassificationError(predictions, labels, k);
        }

        private static Function MSE(Variable labels, Variable predictions)
        {
            return Losses.MeanSquaredError(labels, predictions);
        }
        private static Function MAE(Variable labels, Variable predictions)
        {
            return Losses.MeanAbsError(labels, predictions);
        }

        private static Function MAPE(Variable labels, Variable predictions)
        {
            return Losses.MeanAbsPercentageError(labels, predictions);
        }

        private static Function MSLE(Variable labels, Variable predictions)
        {
            return Losses.MeanSquaredLogError(labels, predictions);
        }

    }
}
