using System;
using System.Collections.Generic;
using System.Text;
using System.Linq;

namespace SiaNet
{
    public class History
    {
        public List<float> TrainLoss { get; set; }

        public List<float> TrainMetric { get; set; }

        public List<float> ValLoss { get; set; }

        public List<float> ValMetric { get; set; }

        public History()
        {
            TrainLoss = new List<float>();
            TrainMetric = new List<float>();
            ValLoss = new List<float>();
            ValMetric = new List<float>();
        }

        public void Add(List<float> trainLoss, List<float> trainMetric, List<float> valLoss, List<float> valMetric)
        {
            TrainLoss.Add(trainLoss.Average());
            TrainMetric.Add(trainMetric.Average());

            if (valLoss.Count > 0)
                ValLoss.Add(valLoss.Average());

            if (valMetric.Count > 0)
                ValMetric.Add(valMetric.Average());
        }
    }
}
