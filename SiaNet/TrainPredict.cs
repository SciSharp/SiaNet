namespace SiaNet
{
    using SiaNet.Data;
    using System;
    using System.Collections.Generic;
    using System.Diagnostics;
    using System.Linq;
    using SiaNet.Engine;
    using SiaNet.Events;

    public partial class Sequential
    {
        /// <summary>
        /// The current backend
        /// </summary>
        internal IBackend K = Global.CurrentBackend;

        /// <summary>
        /// The train losses
        /// </summary>
        private List<float> train_losses = new List<float>();

        /// <summary>
        /// The train metrics
        /// </summary>
        private List<float> train_metrics = new List<float>();

        /// <summary>
        /// The validation losses
        /// </summary>
        private List<float> val_losses = new List<float>();

        /// <summary>
        /// The validation metrics
        /// </summary>
        private List<float> val_metrics = new List<float>();

        /// <summary>
        /// Gets or sets the learning history.
        /// </summary>
        /// <value>
        /// The learning history.
        /// </value>
        public History LearningHistory { get; set; }

        /// <summary>
        /// Occurs when [batch end].
        /// </summary>
        public event EventHandler<BatchEndEventArgs> BatchEnd;

        /// <summary>
        /// Occurs when [on batch start].
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

        /// <summary>
        /// Trains the model for a given number of epochs (iterations on a dataset).
        /// </summary>
        /// <param name="train">The train dataset which is an instance of DataFrame Iter.</param>
        /// <param name="epochs">Integer. Number of epochs to train the model. An epoch is an iteration over the entire x and y data provided. Note that in conjunction with initial_epoch,  epochs is to be understood as "final epoch". The model is not trained for a number of iterations given by epochs, but merely until the epoch of index epochs is reached.</param>
        /// <param name="batchSize">Integer or None. Number of samples per gradient update. If unspecified, batch_size will default to 32.</param>
        /// <param name="val">The validation set of data to evaluate the model at every epoch.</param>
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

        /// <summary>
        /// Runs the epoch.
        /// </summary>
        /// <param name="iteration">The iteration.</param>
        /// <param name="train">The train.</param>
        /// <param name="val">The value.</param>
        /// <returns></returns>
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
                x.Dispose();
                y.Dispose();
            }

            if (val != null)
            {
                while (val.Next())
                {
                    var (x, y) = val.GetBatch();

                    var pred = Forward(x);

                    var lossVal = LossFn.Forward(pred, y);
                    var metricVal = MetricFn.Calc(pred, y);
                    val_losses.Add(K.Mean(lossVal));
                    val_metrics.Add(K.Mean(metricVal));
                    x.Dispose();
                    y.Dispose();
                    lossVal.Dispose();
                    metricVal.Dispose();
                }
            }

            return iteration;
        }

        /// <summary>
        /// Runs the train on batch.
        /// </summary>
        /// <param name="i">The i.</param>
        /// <param name="x">The x.</param>
        /// <param name="y">The y.</param>
        private void RunTrainOnBatch(int i, Tensor x, Tensor y)
        {
            Tensor pred = Forward(x);
            Tensor lossVal = LossFn.Forward(pred, y);
            Tensor grad = LossFn.Backward(pred, y);
            lossVal = ApplyRegularizer(lossVal);
            var metricVal = MetricFn.Calc(pred, y);
            train_losses.Add(K.Mean(lossVal));
            train_metrics.Add(K.Mean(metricVal));

            Backward(grad);

            ApplyDeltaRegularizer();

            foreach (var layer in Layers)
            {
                OptimizerFn.Update(i, layer);
            }

            pred.Dispose();
            lossVal.Dispose();
            grad.Dispose();
        }

        /// <summary>
        /// Generates output predictions for the input samples.
        /// </summary>
        /// <param name="x">The input data frame to run prediction.</param>
        /// <returns></returns>
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

            return K.CreateVariable(predictions.ToArray(), output.Shape);
        }

        /// <summary>
        /// Generates output predictions for the input samples. Computation is done in batches.
        /// </summary>
        /// <param name="x">The input data frame to run prediction.</param>
        /// <param name="batch_size">Size of the batch.</param>
        /// <returns></returns>
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

            return K.CreateVariable(predictions.ToArray(), outshape);
        }

        /// <summary>
        /// Called when [batch end].
        /// </summary>
        /// <param name="epoch">The epoch.</param>
        /// <param name="batch">The batch.</param>
        /// <param name="loss">The loss.</param>
        /// <param name="metric">The metric.</param>
        protected void OnBatchEnd(int epoch, int batch, float loss, float metric)
        {
            BatchEnd?.Invoke(this, new BatchEndEventArgs(epoch, batch, loss, metric));
        }

        /// <summary>
        /// Called when [batch start].
        /// </summary>
        /// <param name="epoch">The epoch.</param>
        /// <param name="batch">The batch.</param>
        protected void OnBatchStart(int epoch, int batch)
        {
            BatchStart?.Invoke(this, new BatchStartEventArgs(epoch, batch));
        }

        /// <summary>
        /// Called when [epoch end].
        /// </summary>
        /// <param name="epoch">The epoch.</param>
        /// <param name="samplesSeenPerSec">The samples seen per sec.</param>
        /// <param name="loss">The loss.</param>
        /// <param name="validationLoss">The validation loss.</param>
        /// <param name="metric">The metric.</param>
        /// <param name="validationMetric">The validation metric.</param>
        /// <param name="duration">The duration.</param>
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

        /// <summary>
        /// Called when [epoch start].
        /// </summary>
        /// <param name="epoch">The epoch.</param>
        protected void OnEpochStart(int epoch)
        {
            EpochStart?.Invoke(this, new EpochStartEventArgs(epoch));
        }

        /// <summary>
        /// Called when [training end].
        /// </summary>
        /// <param name="history">The history.</param>
        /// <param name="duration">The duration.</param>
        protected void OnTrainingEnd(History history, long duration)
        {
            TrainingEnd?.Invoke(this, new TrainingEndEventArgs(history, duration));
        }
    }
}
