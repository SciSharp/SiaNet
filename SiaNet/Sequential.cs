using System;
using System.Collections.Generic;
using System.Text;
using SiaNet.Data;
using SiaNet.Layers;
using SiaNet.Losses;
using SiaNet.Metrics;
using SiaNet.Optimizers;
using System.Linq;
using System.Diagnostics;
using System.Threading;
using System.Threading.Tasks;
using System.IO;
using Newtonsoft.Json;
using SiaNet.Engine;

namespace SiaNet
{
    public partial class Sequential
    {
        public List<BaseLayer> Layers { get; set; }

        public BaseLoss LossFn { get; set; }

        public BaseMetric MetricFn { get; set; }

        public BaseOptimizer OptimizerFn { get; set; }

        public Sequential()
        {
            Layers = new List<BaseLayer>();
        }

        public void Add(BaseLayer l)
        {
            Layers.Add(l);
        }

        private Tensor Forward(Tensor input)
        {
            BaseLayer lastLayer = null;

            foreach (var layer in Layers)
            {
                if (lastLayer == null)
                    layer.Forward(input);
                else
                    layer.Forward(lastLayer.Output);

                lastLayer = layer;
            }

            return lastLayer.Output;
        }

        private void Backward(Tensor gradOutput)
        {
            var curGradOutput = gradOutput;
            for (int i = Layers.Count - 1; i >= 0; --i)
            {
                var layer = Layers[i];

                layer.Backward(curGradOutput);
                curGradOutput = layer.Input.Grad;
            }
        }

        private Tensor ApplyRegularizer(Tensor loss)
        {
            foreach (var l in Layers)
            {
                foreach (var p in l.Params)
                {
                    if(p.Value.HaveRegularizer)
                        loss += p.Value.ApplyRegularizer();
                }
            }

            return loss;
        }

        private void ApplyDeltaRegularizer()
        {
            foreach (var l in Layers)
            {
                foreach (var p in l.Params)
                {
                    if (p.Value.HaveRegularizer)
                        p.Value.ApplyDeltaRegularizer();
                }
            }
        }

        public void Compile(OptimizerType optimizer, LossType loss, MetricType metric)
        {
            OptimizerFn = BaseOptimizer.Get(optimizer);
            LossFn = BaseLoss.Get(loss);
            MetricFn = BaseMetric.Get(metric);
        }

        public void Compile(BaseOptimizer optimizer, LossType loss, MetricType metric)
        {
            OptimizerFn = optimizer;
            LossFn = BaseLoss.Get(loss);
            MetricFn = BaseMetric.Get(metric);
        }

        public void SaveModel(string filePath)
        {
            string modelJson = JsonConvert.SerializeObject(this
                                                    , Formatting.Indented
                                                    , new JsonSerializerSettings() { TypeNameHandling = TypeNameHandling.Auto });
            File.WriteAllText(filePath, modelJson);
        }

        public static Sequential LoadModel(string filePath)
        {
            string jsondata = File.ReadAllText(filePath);
            Sequential model = JsonConvert.DeserializeObject<Sequential>(jsondata, new JsonSerializerSettings() { TypeNameHandling = TypeNameHandling.Auto });
            return model;
        }
    }
}
