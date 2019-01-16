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
    public class Sequential
    {
        public List<BaseLayer> Layers { get; set; }

        private Variable lastOutput;

        public BaseLoss LossFn { get; set; }

        public BaseMetric MetricFn { get; set; }

        public BaseOptimizer OptimizerFn { get; set; }

        List<float> train_losses = new List<float>();

        List<float> train_metrics = new List<float>();

        List<float> val_losses = new List<float>();

        List<float> val_metrics = new List<float>();

        int currentIteration = 0;

        List<IAllocator> contextList = new List<IAllocator>();

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

        public void Fit(IFrameIter train, int epochs, int batchSize, IFrameIter val = null)
        {
            try
            {
                train.SetBatchSize(batchSize);
                List<Task> taskList = new List<Task>();
                for (int iteration = 1; iteration <= epochs; iteration++)
                {
                    RunEpoch(iteration, train);
                }
            }
            catch(Exception ex)
            {
                Console.WriteLine(ex.ToString());
            }
            finally
            {
                GC.Collect();
            }
        }

        private int RunEpoch(int iteration, IFrameIter train, IFrameIter val = null)
        {
            Global.SetNewContext();
            currentIteration = iteration;
            train_losses.Clear();
            train_metrics.Clear();
            val_losses.Clear();
            val_metrics.Clear();
            Stopwatch sw = Stopwatch.StartNew();
            train.Reset();
            List<KeyValuePair<Tensor, Tensor>> batchList = new List<KeyValuePair<Tensor, Tensor>>();
            while (train.Next())
            {
                var (x, y) = train.GetBatch();
               
                
                batchList.Add(new KeyValuePair<Tensor, Tensor>(x, y));
                //Thread t = new Thread(new ParameterizedThreadStart(RunTrainOnBatch));
                //t.Start(train);
            }

            RunTrainOnBatch(batchList[0].Key, batchList[0].Value);

            foreach (var item in batchList.AsParallel().WithDegreeOfParallelism(4))
            {
                RunTrainOnBatch(item.Key, item.Value);
            }

            //Parallel.ForEach(batchList, new ParallelOptions() { MaxDegreeOfParallelism = 4 }, item =>
            //{
                
            //});
            

            if (val != null)
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

            sw.Stop();
            Console.WriteLine("Epoch: {0}, Loss: {1}, Metrics: {2}, Time: {3}(ms)", iteration, train_losses.Average(), train_metrics.Average(), sw.ElapsedMilliseconds);
            return iteration;
        }

        private void RunTrainOnBatch(Tensor x, Tensor y)
        {
            //Global.SetNewContext();
            x = x.ToDevice(Global.Device);
            y = y.ToDevice(Global.Device);
            Variable pred = Forward(x);
            Tensor lossVal = LossFn.Call(pred.Data, y);
            Tensor grad = LossFn.CalcGrad(pred.Data, y);
            Tensor reg_loss = ApplyRegularizer(lossVal);

            var metricVal = MetricFn.Call(pred.Data, y);
            train_losses.Add(TOps.SumF(lossVal));
            train_metrics.Add(TOps.MeanF(metricVal));

            Backward(grad);

            ApplyDeltaRegularizer();

            foreach (var layer in Layers)
            {
                OptimizerFn.Update(currentIteration, layer);
            }
        }
    }
}
