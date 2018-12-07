using System;
using System.Collections.Generic;
using System.Text;

namespace SiaNet.Metrics
{
    public class MetricRegistry
    {
        public static BaseMetric Get(MetricType metricType)
        {
            BaseMetric metric = null;
            switch (metricType)
            {
                case MetricType.Accuracy:
                    metric = new Accuracy();
                    break;
                case MetricType.MeanSquaredError:
                    metric = new MSE();
                    break;
                case MetricType.MeanAbsoluteError:
                    metric = new MAE();
                    break;
                case MetricType.MeanAbsolutePercentageError:
                    metric = new MAPE();
                    break;
                case MetricType.MeanSquareLogError:
                    metric = new MSLE();
                    break;
            }

            return metric;
        }
    }
}
