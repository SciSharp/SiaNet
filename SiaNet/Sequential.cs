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

namespace SiaNet
{
    public class Sequential
    {
        public List<BaseLayer> Layers { get; set; }

        public int[] InputShape { get; set; }

        private Variable lastOutput;

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

        public Sequential(params int[] shape)
        {
            InputShape = shape;
            Layers = new List<BaseLayer>();
        }

        public void Add(BaseLayer l)
        {
            Layers.Add(l);
        }

        public IEnumerable<Variable> GetParameters()
        {
            foreach (var layer in Layers)
            {
                foreach (var v in layer.GetParameters())
                {
                    yield return v;
                }
            }
        }

        private Variable Forward(Tensor input)
        {
            Variable output = input.ToVariable("X");
            foreach (var layer in Layers)
            {
                layer.Forward(output);
                output = layer.Output.ToVariable();
            }

            lastOutput = output;
            return output;
        }

        private Tensor Backward(Tensor gradOutput)
        {
            var curGradOutput = gradOutput;

            for (int i = Layers.Count - 1; i > 0; --i)
            {
                var layer = Layers[i];

                layer.Backward(curGradOutput);
                curGradOutput = layer.Input.Grad;
            }

            Layers[0].Backward(curGradOutput);
            curGradOutput = Layers[0].Input.Grad;
            return curGradOutput;
        }

        private Tensor ApplyRegularizer(TVar loss)
        {
            foreach (var l in Layers)
            {
                foreach (var p in l.Params)
                {
                    if(p.Value.HaveRegularizer)
                        loss += p.Value.ApplyRegularizer();
                }
            }

            return loss.Evaluate();
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

        public void Fit(IFrameIter train, int epochs, int batchSize, IFrameIter val = null)
        {
            DateTime start = DateTime.Now;
            List<float> train_losses = new List<float>();
            List<float> train_metrics = new List<float>();
            List<float> val_losses = new List<float>();
            List<float> val_metrics = new List<float>();

            train.SetBatchSize(batchSize);
            for(int iteration = 1; iteration <= epochs; iteration++)
            {
                train.Reset();
                while (train.Next())
                {
                    var (x, y) = train.GetBatch();

                    using (Variable pred = Forward(x))
                    using (Tensor lossVal = LossFn.Call(pred.Data, y))
                    using (Tensor grad = LossFn.CalcGrad(pred.Data, y))
                    using (Tensor reg_loss = ApplyRegularizer(lossVal))
                    {
                        //var metricVal = MetricFn.Call(pred.Data, y);
                        train_losses.Add(reg_loss.TVar().ToScalar().Evaluate());
                        //train_metrics.Add(metricVal.ToScalar().Evaluate());

                        Backward(grad);

                        ApplyDeltaRegularizer();

                        foreach (var layer in Layers)
                        {
                            OptimizerFn.Update(iteration, layer);
                        }
                    }

                    x.Dispose();
                    y.Dispose();
                }

                if(val != null)
                {
                    while (val.Next())
                    {
                        var (x, y) = val.GetBatch();

                        var pred = Forward(x);

                        var lossVal = LossFn.Call(pred.Data, y);
                        var metricVal = MetricFn.Call(pred.Data, y);
                        val_losses.Add(TOps.MeanF(lossVal));
                        val_metrics.Add(TOps.MeanF(metricVal));
                    }
                }

                Console.WriteLine("Epoch: {0}, Loss: {1}", iteration, train_losses.Average());
            }
        }
    }
}
