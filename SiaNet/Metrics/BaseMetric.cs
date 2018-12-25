using System;
using System.Collections.Generic;
using System.Text;
using TensorSharp.Expression;

namespace SiaNet.Metrics
{
    public abstract class BaseMetric
    {
        public string Name { get; set; }

        public BaseMetric(string name)
        {
            Name = name;
        }

        public abstract TVar Call(TVar preds, TVar labels);

        public static BaseMetric Get(MetricType metricType)
        {
            BaseMetric metric = null;
            switch (metricType)
            {
                case MetricType.Accuracy:
                    metric = new Accuracy();
                    break;
                case MetricType.MSE:
                    metric = new MSE();
                    break;
                case MetricType.MAE:
                    metric = new MAE();
                    break;
                case MetricType.MAPE:
                    metric = new MAPE();
                    break;
                case MetricType.MSLE:
                    metric = new MSLE();
                    break;
                default:
                    break;
            }

            return metric;
        }
    }
}
