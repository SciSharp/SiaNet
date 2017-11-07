using CNTK;
using SiaNet.Events;
using SiaNet.Interface;
using SiaNet.Model;
using System.Collections.Generic;
using System.Linq;

namespace SiaNet.Processing
{
    internal class ImgGenTrainPredict : ITrainPredict
    {
        public Function Model;
        private Function lossFunc;
        private Function metricFunc;
        private IList<Learner> learners;
        private Variable featureVariable;
        private Variable labelVariable;
        private string lossName;
        private string metricName;

        public ImgGenTrainPredict(Function model, Function lossFunc, string lossName, Function metricFunc, string metricName, IList<Learner> learners, Variable featureVariable, Variable labelVariable)
        {
            this.Model = model;
            this.lossFunc = lossFunc;
            this.metricFunc = metricFunc;
            this.learners = learners;
            this.featureVariable = featureVariable;
            this.labelVariable = labelVariable;
            this.metricName = metricName;
            this.lossName = lossName;
        }

        public Dictionary<string, List<double>> Train(object trainData, object validationData, int epoches, int batchSize, On_Epoch_Start OnEpochStart, On_Epoch_End OnEpochEnd)
        {
            ImageDataGenerator train = (ImageDataGenerator)trainData;
            ImageDataGenerator validation = validationData != null ? (ImageDataGenerator)validationData : null;
            Dictionary<string, List<double>> result = new Dictionary<string, List<double>>();
            var trainer = Trainer.CreateTrainer(Model, lossFunc, metricFunc, learners);
            int currentEpoch = 1;
            Dictionary<string, double> metricsList = new Dictionary<string, double>();
            int imageSize = featureVariable.Shape.Rank == 1 ? featureVariable.Shape[0] : featureVariable.Shape[0] * featureVariable.Shape[1] * featureVariable.Shape[2];
            int numClasses = labelVariable.Shape[0];
            IList<StreamConfiguration> streamConfigurations = new StreamConfiguration[] { new StreamConfiguration("features", imageSize), new StreamConfiguration("labels", numClasses) };
            List<double> totalBatchLossList = new List<double>();
            List<double> totalMetricValueList = new List<double>();
            if (train.GenType == ImageGenType.FromTextFile)
            {
                train.LoadTextData(featureVariable, labelVariable);
                if(validation != null)
                    validation.LoadTextData(featureVariable, labelVariable);
            }

            while (currentEpoch <= epoches)
            {
                metricsList.Clear();
                totalBatchLossList.Clear();
                totalMetricValueList.Clear();
                OnEpochStart(currentEpoch);

                while (!train.NextBatch(batchSize))
                {
                    trainer.TrainMinibatch(new Dictionary<Variable, MinibatchData> { { featureVariable, train.CurrentBatchX }, { labelVariable, train.CurrentBatchY } }, GlobalParameters.Device);
                    totalBatchLossList.Add(trainer.PreviousMinibatchLossAverage());
                    totalMetricValueList.Add(trainer.PreviousMinibatchEvaluationAverage());
                }

                if (!result.ContainsKey("loss"))
                {
                    result.Add("loss", new List<double>());
                }

                if (!result.ContainsKey(metricName))
                {
                    result.Add(metricName, new List<double>());
                }

                double lossValue = totalBatchLossList.Average();
                double metricValue = totalMetricValueList.Average();
                result["loss"].Add(lossValue);
                result[metricName].Add(metricValue);
                metricsList.Add(metricName, metricValue);
                if (validation != null)
                {
                    if (!result.ContainsKey("val_loss"))
                    {
                        result.Add("val_loss", new List<double>());
                    }

                    if (!result.ContainsKey("val_" + metricName))
                    {
                        result.Add("val_" + metricName, new List<double>());
                    }

                    List<double> totalEvalBatchLossList = new List<double>();
                    List<double> totalEvalMetricValueList = new List<double>();
                    while (validation.NextBatch(batchSize))
                    {
                        Variable actualVariable = CNTKLib.InputVariable(labelVariable.Shape, DataType.Float);
                        var evalLossFunc = Losses.Get(lossName, labelVariable, actualVariable);
                        var evalMetricFunc = Metrics.Get(metricName, labelVariable, actualVariable);
                        Value actual = EvaluateInternal(validation.CurrentBatchX.data);
                        Value expected = validation.CurrentBatchY.data;
                        var inputDataMap = new Dictionary<Variable, Value>() { { labelVariable, expected }, { actualVariable, actual } };
                        var outputDataMap = new Dictionary<Variable, Value>() { { evalLossFunc.Output, null } };

                        evalLossFunc.Evaluate(inputDataMap, outputDataMap, GlobalParameters.Device);
                        var evalLoss = outputDataMap[evalLossFunc.Output].GetDenseData<float>(evalLossFunc.Output).Select(x => x.First()).ToList();
                        totalEvalBatchLossList.Add(evalLoss.Average());

                        inputDataMap = new Dictionary<Variable, Value>() { { labelVariable, expected }, { actualVariable, actual } };
                        outputDataMap = new Dictionary<Variable, Value>() { { evalMetricFunc.Output, null } };
                        evalMetricFunc.Evaluate(inputDataMap, outputDataMap, GlobalParameters.Device);
                        var evalMetric = outputDataMap[evalMetricFunc.Output].GetDenseData<float>(evalMetricFunc.Output).Select(x => x.First()).ToList();
                        totalEvalMetricValueList.Add(evalMetric.Average());
                    }

                    result["val_loss"].Add(totalEvalBatchLossList.Average());
                    metricsList.Add("val_loss", totalEvalBatchLossList.Average());
                    result["val_" + metricName].Add(totalEvalMetricValueList.Average());
                    metricsList.Add("val_" + metricName, totalEvalMetricValueList.Average());
                }

                OnEpochEnd(currentEpoch, trainer.TotalNumberOfSamplesSeen(), lossValue, metricsList);
                currentEpoch++;
            }

            return result;
        }

        public static bool MiniBatchDataIsSweepEnd(ICollection<MinibatchData> minibatchValues)
        {
            return minibatchValues.Any(a => a.sweepEnd);
        }

        

        private Value EvaluateInternal(Value data)
        {
            var outputDataMap = new Dictionary<Variable, Value>() { { Model.Output, null } };
            Model.Evaluate(new Dictionary<Variable, Value>() { { featureVariable, data } }, outputDataMap, GlobalParameters.Device);
            return outputDataMap[Model.Output];
        }

        public IList<float> Evaluate(object testData)
        {
            Value data = (Value)testData;
            var outputValue = EvaluateInternal(data);
            IList<IList<float>> resultSet = outputValue.GetDenseData<float>(Model.Output);
            var result = resultSet.Select(x => x.First()).ToList();
            return result;
        }
    }
}
