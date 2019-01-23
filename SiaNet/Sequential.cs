using System;
using System.Collections.Generic;
using System.Text;
using SiaNet.Data;
using SiaNet.Layers;
using SiaNet.Losses;
using SiaNet.Metrics;
using SiaNet.Optimizers;
using TensorSharp.Expression;
using System.Linq;
using TensorSharp;
using System.Diagnostics;
using System.Threading;
using TensorSharp.CUDA;
using System.Threading.Tasks;

namespace SiaNet
{
    public partial class Sequential
    {
        public List<BaseLayer> Layers { get; set; }

        public BaseLoss LossFn { get; set; }

        public BaseMetric MetricFn { get; set; }

        public BaseOptimizer OptimizerFn { get; set; }

        public long TotalParameterCount
        {
            get
            {
                var parameters = GetParameters();
                long total = 0;
                foreach (var item in parameters)
                {
                    total += item.Data.ElementCount();
                }

                return total;
            }
        }

        public Sequential()
        {
            Layers = new List<BaseLayer>();
        }

        public void Add(BaseLayer l)
        {
            Layers.Add(l);
        }

        public IEnumerable<Parameter> GetParameters()
        {
            foreach (var layer in Layers)
            {
                foreach (var v in layer.GetParameters())
                {
                    yield return v;
                }
            }
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
    }
}
