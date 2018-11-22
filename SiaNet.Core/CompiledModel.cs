using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using CNTK;
using SiaNet.EventArgs;
using SiaNet.Data;
using SiaNet.Metrics;
using SiaNet.Optimizers;
using Function = CNTK.Function;
using Variable = CNTK.Variable;

namespace SiaNet
{
    public class CompiledModel : IDisposable
    {
        /// <summary>
        ///     Occurs when [on batch end].
        /// </summary>
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

        /// <summary>
        ///     Occurs when [on training start].
        /// </summary>
        public event EventHandler TrainingStart;

        protected readonly Variable FeatureVariable;
        protected readonly Variable LabelVariable;
        protected readonly Function Model;

        public CompiledModel(Function model)
        {
            Model = model;
            LabelVariable = Variable.InputVariable(new[] {Model.Output.Shape[0]}, DataType.Float);
            FeatureVariable = Model.Inputs.FirstOrDefault(variable => variable.IsInput);
        }

        public CompiledModel Clone()
        {
            return new CompiledModel(this.Model.Clone(ParameterCloningMethod.Clone));
        }

        public NDArrayView[] GetParameters()
        {
            return Model.Parameters().Select(parameter => parameter.GetValue()).ToArray();
        }

        public void SetParameters(NDArrayView[] values)
        {
            var parameters = Model.Parameters();

            if (parameters.Count != values.Length)
            {
                throw new ArgumentException();
            }

            for (int i = 0; i < parameters.Count; i++)
            {
                parameters[i].SetValue(values[i]);
            }
        }

        public Shape InputShape
        {
            get => FeatureVariable.Shape;
        }

        public Shape OutputShape
        {
            get => LabelVariable.Shape;
        }

        /// <inheritdoc />
        public void Dispose()
        {
            FeatureVariable?.Dispose();
            LabelVariable?.Dispose();
            Model?.Dispose();
        }

        public static CompiledModel Load(string modelFilename)
        {
            return new CompiledModel(Function.Load(modelFilename, GlobalParameters.Device));
        }

        public static CompiledModel Load(byte[] binaryModel)
        {
            return new CompiledModel(Function.Load(binaryModel, GlobalParameters.Device));
        }

        public static CompiledModel Load(Stream modelStream)
        {
            using (var memoryStream = new MemoryStream())
            {
                modelStream.CopyTo(memoryStream);

                return Load(memoryStream.ToArray());
            }
        }

        public double Evaluate(
            IDataFrameList validationData,
            int batchSize,
            MetricFunction lossMetric)
        {
            return Evaluate(validationData, batchSize, lossMetric, null, out _);
        }

        public double Evaluate(
            IDataFrameList validationData,
            int batchSize,
            MetricFunction lossMetric,
            MetricFunction evaluationMetric,
            out double evaluationResult)
        {
            var losses = new List<double>();
            var metrics = new List<double>();

            using (var actualVariable = CNTKLib.InputVariable(LabelVariable.Shape, DataType.Float))
            using (var lossFunction = lossMetric.ToFunction(LabelVariable, actualVariable))
            {
                var metricFunction = evaluationMetric?.ToFunction(LabelVariable, actualVariable);
                var batchId = 1;
                IDataFrameList currentBatch;
                while ((currentBatch = validationData.ToBatch(batchId, batchSize)) != null)
                {
                    using (var actual = Evaluate(currentBatch.Features.ToValue()))
                    using (var expected = currentBatch.Labels.ToValue())
                    {
                        var inputDataMap =
                            new Dictionary<Variable, Value> {{LabelVariable, expected}, {actualVariable, actual}};
                        var outputDataMap = new Dictionary<Variable, Value> {{lossFunction.Output, null}};

                        lossFunction.Evaluate(inputDataMap, outputDataMap, GlobalParameters.Device);
                        var batchLoss = outputDataMap[lossFunction.Output].GetDenseData<float>(lossFunction.Output)
                            .Select(x => x.First()).Average();
                        losses.Add(batchLoss);

                        if (metricFunction != null)
                        {
                            outputDataMap = new Dictionary<Variable, Value> {{metricFunction.Output, null}};

                            metricFunction.Evaluate(inputDataMap, outputDataMap, GlobalParameters.Device);
                            var batchMetric = outputDataMap[metricFunction.Output]
                                .GetDenseData<float>(metricFunction.Output).Select(x => x.First()).Average();
                            metrics.Add(batchMetric);
                        }
                    }

                    batchId++;
                }

                metricFunction?.Dispose();
            }

            var loss = losses.Average();
            evaluationResult = metrics.Any() ? metrics.Average() : loss;

            return loss;
        }
        
        /// <summary>
        ///     Fits the model for a fixed number of epochs.
        /// </summary>
        /// <param name="trainData">The training dataset.</param>
        /// <param name="epoches">The no. of trainin epoches.</param>
        /// <param name="batchSize">Size of the batch for training.</param>
        /// <param name="validation">The validation dataset.</param>
        /// <param name="shuffle">Shuffle the dataset while training</param>
        public void Fit(
            IDataFrameList trainData,
            uint epoches,
            int batchSize,
            OptimizerBase optimizer,
            MetricFunction lossMetric,
            MetricFunction evaluationMetric = null,
            IDataFrameList validation = null,
            bool shuffle = false)
        {
            var lastEpochLoss = 0d;
            var lastEpochMetric = 0d;
            var lastEvaluationLoss = 0d;
            var lastEvaluationMetric = 0d;

            using (var lossFunction = lossMetric.ToFunction(LabelVariable, (Data.Function) Model))
            using (var evaluationFunction =
                (evaluationMetric ?? lossMetric).ToFunction(LabelVariable, (Data.Function) Model))
            using (var learner = optimizer.ToLearner(Model))
            using (var trainer = CNTK.Trainer.CreateTrainer(Model, lossFunction, evaluationFunction, new[] {learner}))
            {
                OnTrainingStart();
                var currentEpoch = 1u;

                while (currentEpoch <= epoches)
                {
                    if (shuffle)
                    {
                        trainData.Shuffle();
                    }

                    OnEpochStart(currentEpoch);
                    var batchId = 1;
                    var epochLosses = new List<double>();
                    var epochMetrics = new List<double>();
                    IDataFrameList currentBatch;
                    while ((currentBatch = trainData.ToBatch(batchId - 1, batchSize)) != null)
                    {
                        OnBatchStart(currentEpoch, batchId);

                        using (var features = currentBatch.Features.ToValue())
                        using (var labels = currentBatch.Labels.ToValue())
                        {
                            trainer.TrainMinibatch(
                                new Dictionary<Variable, Value>
                                {
                                    {FeatureVariable, features},
                                    {LabelVariable, labels}
                                }, false,
                                GlobalParameters.Device);
                        }

                        var batchLoss = trainer.PreviousMinibatchLossAverage();
                        var batchMetric = trainer.PreviousMinibatchEvaluationAverage();
                        epochLosses.Add(batchLoss);
                        epochMetrics.Add(batchMetric);

                        OnBatchEnd(currentEpoch, batchId, trainer.TotalNumberOfSamplesSeen(), batchLoss,
                            batchMetric);
                        batchId++;
                    }

                    lastEpochLoss = epochLosses.Average();
                    lastEpochMetric = epochMetrics.Average();

                    if (validation != null)
                    {
                        lastEvaluationLoss = Evaluate(validation, batchSize, lossMetric, evaluationMetric,
                            out lastEvaluationMetric);
                    }

                    OnEpochEnd(currentEpoch, trainer.TotalNumberOfSamplesSeen(), lastEpochLoss, lastEvaluationLoss,
                        lastEpochMetric, lastEvaluationMetric);
                    currentEpoch++;
                }
            }

            GC.Collect();
            OnTrainingEnd(lastEpochLoss, lastEvaluationLoss, lastEpochMetric, lastEvaluationMetric);
        }

        /// <summary>
        ///     Predicts the specified data.
        /// </summary>
        /// <param name="data">The data for prediction.</param>
        /// <returns>List of prediction values</returns>
        public DataFrame Predict(IDataFrame data, int batchSize = 64)
        {
            var temporaryDataFrameList = new DataFrameList(data.DataShape, 1);

            for (int i = 0; i < data.Length; i++)
            {
                temporaryDataFrameList.AddFrame(data[i], 0f);
            }

            var batchId = 0;
            IDataFrameList currentBatch;
            var df = new DataFrame(OutputShape);
            while ((currentBatch = temporaryDataFrameList.ToBatch(batchId, batchSize)) != null)
            {
                var outputValue = Evaluate(currentBatch.Features.ToValue());
                var resultSet = outputValue.GetDenseData<float>(Model.Output);

                foreach (var result in resultSet)
                {
                    df.Add(result.ToArray());
                }
                batchId++;
            }
            
            return df;
        }

        public float[] Predict(float[] data)
        {
            var dataFrame = new DataFrame(data.Length);
            dataFrame.Add(data);
            return Predict(dataFrame)[0];
        }

        public void Save(string modelFilename)
        {
            Model.Save(modelFilename);
        }

        public void Save(Stream modelStream)
        {
            var modelBytes = Model.Save();
            modelStream.Write(modelBytes, 0, modelBytes.Length);
        }

        protected Value Evaluate(Value data)
        {
            using (var features = data)
            {
                var inputDataMap = new Dictionary<Variable, Value> {{FeatureVariable, features}};
                var outputDataMap = new Dictionary<Variable, Value> {{Model.Output, null}};
                Model.Evaluate(inputDataMap, outputDataMap, GlobalParameters.Device);
                return outputDataMap[Model.Output];
            }
        }

        protected void OnBatchEnd(uint epoch, int batch, ulong samplesSeen, double loss, double metric)
        {
            BatchEnd?.Invoke(this, new BatchEndEventArgs(epoch, batch, samplesSeen, loss, metric));
        }

        protected void OnBatchStart(uint epoch, int batch)
        {
            BatchStart?.Invoke(this, new BatchStartEventArgs(epoch, batch));
        }

        protected void OnEpochEnd(
            uint epoch,
            ulong samplesSeen,
            double loss,
            double validationLoss,
            double metric,
            double validationMetric)
        {
            EpochEnd?.Invoke(this,
                new EpochEndEventArgs(epoch, samplesSeen, loss, validationLoss, metric, validationMetric));
        }

        protected void OnEpochStart(uint epoch)
        {
            EpochStart?.Invoke(this, new EpochStartEventArgs(epoch));
        }

        protected void OnTrainingEnd(double loss, double validationLoss, double metric, double validationMetric)
        {
            TrainingEnd?.Invoke(this, new TrainingEndEventArgs(loss, validationLoss, metric, validationMetric));
        }

        protected void OnTrainingStart()
        {
            TrainingStart?.Invoke(this, new System.EventArgs());
        }
    }
}