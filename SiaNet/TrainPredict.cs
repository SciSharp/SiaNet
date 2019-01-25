using SiaNet.Data;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Text;
using TensorSharp;
using System.Linq;

namespace SiaNet
{
    public partial class Sequential
    {
        List<float> train_losses = new List<float>();

        List<float> train_metrics = new List<float>();

        List<float> val_losses = new List<float>();

        List<float> val_metrics = new List<float>();

        public History LearningHistory { get; set; }

        public event EventHandler<BatchEndEventArgs> BatchEnd;

        /// <summary>
        ///     Occurs when [on batch start].
        /// </summary>
        public event EventHandler<BatchStartEventArgs> BatchStart;

        /// <summary>
        ///     Occurs when [on epoch end].
        /// </summary>
        public event EventHandler<EpochEndEventArgs> EpochEnd;

        /// <summary>
        ///     Occurs when [on epoch start].
        /// </summary>
        public event EventHandler<EpochStartEventArgs> EpochStart;

        /// <summary>
        ///     Occurs when [on training end].
        /// </summary>
        public event EventHandler<TrainingEndEventArgs> TrainingEnd;

        public void Train(DataFrameIter train, int epochs, int batchSize, DataFrameIter val = null)
        {
            LearningHistory = new History();
            Stopwatch trainWatch = new Stopwatch();
            try
            {
                Stopwatch batchWatch = new Stopwatch();
                
                long n = train.DataSize;
                trainWatch.Start();
                train.SetBatchSize(batchSize);
                if (val != null)
                    val.SetBatchSize(batchSize);
                for (int iteration = 1; iteration <= epochs; iteration++)
                {
                    batchWatch.Restart();

                    OnEpochStart(iteration);
                    RunEpoch(iteration, train, val);

                    batchWatch.Stop();
                    long samplesSeen = n * 1000 / (batchWatch.ElapsedMilliseconds + 1);
                    if(val == null)
                        OnEpochEnd(iteration, samplesSeen, train_losses.Average(), 0, train_metrics.Average(), 0, batchWatch.ElapsedMilliseconds);
                    else
                        OnEpochEnd(iteration, samplesSeen, train_losses.Average(), val_losses.Average(), train_metrics.Average(), val_metrics.Average(), batchWatch.ElapsedMilliseconds);

                    LearningHistory.Add(train_losses, train_metrics, val_losses, val_metrics);
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.ToString());
            }

            trainWatch.Stop();
            OnTrainingEnd(LearningHistory, trainWatch.ElapsedMilliseconds);
        }

        private int RunEpoch(int iteration, DataFrameIter train, DataFrameIter val = null)
        {
            train_losses.Clear();
            train_metrics.Clear();
            val_losses.Clear();
            val_metrics.Clear();

            train.Reset();
            if (val != null)
                val.Reset();

            while (train.Next())
            {
                var (x, y) = train.GetBatch();
                RunTrainOnBatch(iteration, x, y);
            }

            if (val != null)
            {
                while (val.Next())
                {
                    var (x, y) = val.GetBatch();

                    var pred = Forward(x);

                    var lossVal = LossFn.Call(pred, y);
                    var metricVal = MetricFn.Call(pred, y);
                    val_losses.Add(TOps.SumF(lossVal));
                    val_metrics.Add(TOps.MeanF(metricVal));
                }
            }

            return iteration;
        }

        private void RunTrainOnBatch(int i, Tensor x, Tensor y)
        {
            Tensor pred = Forward(x);
            Tensor lossVal = LossFn.Call(pred, y);
            Tensor grad = LossFn.CalcGrad(pred, y);
            lossVal = ApplyRegularizer(lossVal);

            var metricVal = MetricFn.Call(pred, y);
            train_losses.Add(TOps.SumF(lossVal));
            train_metrics.Add(TOps.MeanF(metricVal));

            Backward(grad);

            ApplyDeltaRegularizer();

            foreach (var layer in Layers)
            {
                OptimizerFn.Update(i, layer);
            }
        }

        public Tensor Predict(DataFrame x)
        {
            List<float> predictions = new List<float>();

            Tensor output = x.GetTensor();
            foreach (var layer in Layers)
            {
                if (layer.SkipPred)
                    continue;

                layer.Forward(output);
                output = layer.Output;
            }

            predictions.AddRange(output.ToArray().Cast<float>());
            Tensor pred = Tensor.FromArray(Global.Device, predictions.ToArray());
            
            return pred.View(output.Shape);
        }

        public Tensor Predict(DataFrame x, int batch_size)
        {
            DataFrameIter dataFrameIter = new DataFrameIter(x);
            List<float> predictions = new List<float>();
            dataFrameIter.SetBatchSize(batch_size);
            long[] outshape = null;

            while (dataFrameIter.Next())
            {
                var data = dataFrameIter.GetBatchX();
                Tensor output = data;
                foreach (var layer in Layers)
                {
                    if (layer.SkipPred)
                        continue;

                    layer.Forward(output);
                    output = layer.Output;
                }

                predictions.AddRange(output.ToArray().Cast<float>());
            }

            Tensor pred = new Tensor(Global.Device, DType.Float32, outshape);
            pred.CopyFrom(predictions.ToArray());

            return pred;
        }

        protected void OnBatchEnd(int epoch, int batch, float loss, float metric)
        {
            BatchEnd?.Invoke(this, new BatchEndEventArgs(epoch, batch, loss, metric));
        }

        protected void OnBatchStart(int epoch, int batch)
        {
            BatchStart?.Invoke(this, new BatchStartEventArgs(epoch, batch));
        }

        protected void OnEpochEnd(
            int epoch,
            long samplesSeenPerSec,
            float loss,
            float validationLoss,
            float metric,
            float validationMetric,
            long duration)
        {
            EpochEnd?.Invoke(this,
                new EpochEndEventArgs(epoch, samplesSeenPerSec, loss, validationLoss, metric, validationMetric, duration));
        }

        protected void OnEpochStart(int epoch)
        {
            EpochStart?.Invoke(this, new EpochStartEventArgs(epoch));
        }

        protected void OnTrainingEnd(History history, long duration)
        {
            TrainingEnd?.Invoke(this, new TrainingEndEventArgs(history, duration));
        }
    }
}
