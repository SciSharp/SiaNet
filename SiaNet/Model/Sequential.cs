using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Newtonsoft.Json;
using System.IO;
using CNTK;
using SiaNet.Model.Layers;
using System.Data;

namespace SiaNet.Model
{
    public delegate void On_Training_Start();

    public delegate void On_Training_End(Dictionary<string, List<double>> trainingResult);

    public delegate void On_Epoch_Start(int epoch);

    public delegate void On_Epoch_End(int epoch, uint samplesSeen, double loss, Dictionary<string, double> metrics);

    public class Sequential : ConfigModule
    {
        public event On_Training_Start OnTrainingStart;

        public event On_Training_End OnTrainingEnd;

        public event On_Epoch_Start OnEpochStart;

        public event On_Epoch_End OnEpochEnd;

        public string Module
        {
            get
            {
                return "Sequential";
            }
        }

        private List<Learner> learners;

        private Function lossFunc;

        private Function metricFunc;

        private Function modelOut;

        private string metricName;

        private string lossName;

        private bool isConvolution;

        private Variable featureVariable;

        private Variable labelVariable;

        public Dictionary<string, List<double>> TrainingResult { get; set; }

        public List<LayerConfig> Layers { get; set; }

        public Sequential()
        {
            OnTrainingStart += Sequential_OnTrainingStart;
            OnTrainingEnd += Sequential_OnTrainingEnd;
            OnEpochStart += Sequential_OnEpochStart;
            OnEpochEnd += Sequential_OnEpochEnd;
            TrainingResult = new Dictionary<string, List<double>>();
            Layers = new List<LayerConfig>();
            learners = new List<Learner>();
        }

        private void Sequential_OnEpochEnd(int epoch, uint samplesSeen, double loss, Dictionary<string, double> metrics)
        {
            
        }

        private void Sequential_OnEpochStart(int epoch)
        {
            
        }

        private void Sequential_OnTrainingEnd(Dictionary<string, List<double>> trainingResult)
        {
            
        }

        private void Sequential_OnTrainingStart()
        {
            
        }

        public static Sequential LoadNetConfig(string filepath)
        {
            string json = File.ReadAllText(filepath);
            var result = JsonConvert.DeserializeObject<Sequential>(json);

            return result;
        }

        public void SaveNetConfig(string filepath)
        {
            string json = JsonConvert.SerializeObject(this, Formatting.Indented);
            File.WriteAllText(filepath, json);
        }

        public void SaveModel(string filepath)
        {
            modelOut.Save(filepath);
        }

        public void LoadModel(string filepath)
        {
            modelOut = Function.Load(filepath, GlobalParameters.Device);
        }

        public void Add(LayerConfig config)
        {
            Layers.Add(config);
        }

        public void Compile(string optimizer, string loss, string metric)
        {
            CompileModel();
            learners.Add(Optimizers.Get(optimizer, modelOut));
            metricName = metric;
            lossName = loss;
            lossFunc = Losses.Get(loss, labelVariable, modelOut);
            metricFunc = Metrics.Get(metric, labelVariable, modelOut);
        }

        public void Compile(Learner optimizer, string loss, string metric)
        {
            CompileModel();
            learners.Add(optimizer);
            metricName = metric;
            lossName = loss;
            lossFunc = Losses.Get(loss, labelVariable, modelOut);
            metricFunc = Metrics.Get(metric, labelVariable, modelOut);
        }

        private void CompileModel()
        {
            bool first = true;
            foreach (var item in Layers)
            {
                if (first)
                {
                    BuildFirstLayer(item);
                    first = false;
                    continue;
                }

                BuildStackedLayer(item);
            }

            int outputNums = modelOut.Output.Shape[0];
            labelVariable = Variable.InputVariable(new int[] { outputNums }, DataType.Float);
        }

        private void BuildStackedLayer(LayerConfig layer)
        {
            switch (layer.Name.ToUpper())
            {
                case OptLayers.Dense:
                    var l1 = (Dense)layer;
                    modelOut = NN.Basic.Dense(modelOut, l1.Dim, l1.Act, l1.UseBias, l1.WeightInitializer, l1.BiasInitializer);
                    break;
                case OptLayers.Activation:
                    var l2 = (Activation)layer;
                    modelOut = NN.Basic.Activation(modelOut, l2.Act);
                    break;
                case OptLayers.Dropout:
                    var l3 = (Dropout)layer;
                    modelOut = NN.Basic.Dropout(modelOut, l3.Rate);
                    break;
                case OptLayers.BatchNorm:
                    var l4 = (BatchNorm)layer;
                    modelOut = NN.Basic.BatchNorm(modelOut, l4.Epsilon, l4.BetaInitializer, l4.GammaInitializer, l4.RunningMeanInitializer, l4.RunningStdInvInitializer, l4.Spatial, l4.NormalizationTimeConstant, l4.BlendTimeConst);
                    break;
                case OptLayers.Conv1D:
                    var l5 = (Conv1D)layer;
                    modelOut = NN.Convolution.Conv1D(modelOut, l5.Channels, l5.KernalSize, l5.Strides, l5.Padding, l5.Dialation, l5.Act, l5.UseBias, l5.WeightInitializer, l5.BiasInitializer);
                    break;
                case OptLayers.Conv2D:
                    var l6 = (Conv2D)layer;
                    modelOut = NN.Convolution.Conv2D(modelOut, l6.Channels, l6.KernalSize, l6.Strides, l6.Padding, l6.Dialation, l6.Act, l6.UseBias, l6.WeightInitializer, l6.BiasInitializer);
                    break;
                case OptLayers.Conv3D:
                    var l7 = (Conv3D)layer;
                    modelOut = NN.Convolution.Conv3D(modelOut, l7.Channels, l7.KernalSize, l7.Strides, l7.Padding, l7.Dialation, l7.Act, l7.UseBias, l7.WeightInitializer, l7.BiasInitializer);
                    break;
                case OptLayers.MaxPool1D:
                    var l8 = (MaxPool1D)layer;
                    modelOut = NN.Convolution.MaxPool1D(modelOut, l8.PoolSize, l8.Strides, l8.Padding);
                    break;
                case OptLayers.MaxPool2D:
                    var l9 = (MaxPool2D)layer;
                    modelOut = NN.Convolution.MaxPool2D(modelOut, l9.PoolSize, l9.Strides, l9.Padding);
                    break;
                case OptLayers.MaxPool3D:
                    var l10 = (MaxPool3D)layer;
                    modelOut = NN.Convolution.MaxPool3D(modelOut, l10.PoolSize, l10.Strides, l10.Padding);
                    break;
                case OptLayers.AvgPool1D:
                    var l11 = (AvgPool1D)layer;
                    modelOut = NN.Convolution.AvgPool1D(modelOut, l11.PoolSize, l11.Strides, l11.Padding);
                    break;
                case OptLayers.AvgPool2D:
                    var l12 = (AvgPool2D)layer;
                    modelOut = NN.Convolution.AvgPool2D(modelOut, l12.PoolSize, l12.Strides, l12.Padding);
                    break;
                case OptLayers.AvgPool3D:
                    var l113 = (AvgPool3D)layer;
                    modelOut = NN.Convolution.AvgPool3D(modelOut, l113.PoolSize, l113.Strides, l113.Padding);
                    break;
                case OptLayers.GlobalMaxPool1D:
                    modelOut = NN.Convolution.GlobalMaxPool1D(modelOut);
                    break;
                case OptLayers.GlobalMaxPool2D:
                    modelOut = NN.Convolution.GlobalMaxPool2D(modelOut);
                    break;
                case OptLayers.GlobalMaxPool3D:
                    modelOut = NN.Convolution.GlobalMaxPool3D(modelOut);
                    break;
                case OptLayers.GlobalAvgPool1D:
                    modelOut = NN.Convolution.GlobalAvgPool1D(modelOut);
                    break;
                case OptLayers.GlobalAvgPool2D:
                    modelOut = NN.Convolution.GlobalAvgPool2D(modelOut);
                    break;
                case OptLayers.GlobalAvgPool3D:
                    modelOut = NN.Convolution.GlobalAvgPool3D(modelOut);
                    break;
                default:
                    throw new InvalidOperationException(string.Format("{0} layer is not implemented."));
            }
        }

        private void BuildFirstLayer(LayerConfig layer)
        {
            isConvolution = false;
            switch (layer.Name.ToUpper())
            {
                case OptLayers.Dense:
                    var l1 = (Dense)layer;
                    if (!l1.Shape.HasValue)
                        throw new ArgumentNullException("Input shape is missing for first layer");
                    featureVariable = Variable.InputVariable(new int[] { l1.Shape.Value }, DataType.Float);
                    modelOut = NN.Basic.Dense(featureVariable, l1.Dim, l1.Act, l1.UseBias, l1.WeightInitializer, l1.BiasInitializer);
                    break;
                case OptLayers.BatchNorm:
                    var l2 = (BatchNorm)layer;
                    if (!l2.Shape.HasValue)
                        throw new ArgumentNullException("Input shape is missing for first layer");
                    break;
                default:
                    throw new InvalidOperationException(string.Format("{0} cannot be used as first layer."));
            }
        }

        public void Train(XYFrame train, int batchSize, int epoches, XYFrame test = null)
        {
            OnTrainingStart();
            var trainer = Trainer.CreateTrainer(modelOut, lossFunc, metricFunc, learners);
            int currentEpoch = 1;
            Dictionary<string, double> metricsList = new Dictionary<string, double>();
            while (currentEpoch <= epoches)
            {
                metricsList.Clear();
                OnEpochStart(currentEpoch);
                int miniBatchCount = 1;
                List<double> totalBatchLossList = new List<double>();
                List<double> totalMetricValueList = new List<double>();
                while (train.NextBatch(miniBatchCount, batchSize))
                {
                    Value features = GetValueBatch(train.CurrentBatch.XFrame);
                    Value labels = GetValueBatch(train.CurrentBatch.YFrame);
                    
                    trainer.TrainMinibatch(new Dictionary<Variable, Value>() { { featureVariable, features }, { labelVariable, labels } }, GlobalParameters.Device);
                    totalBatchLossList.Add(trainer.PreviousMinibatchLossAverage());
                    totalMetricValueList.Add(trainer.PreviousMinibatchEvaluationAverage());
                    miniBatchCount++;
                }

                if (!TrainingResult.ContainsKey("loss"))
                {
                    TrainingResult.Add("loss", new List<double>());
                }

                if (!TrainingResult.ContainsKey(metricName))
                {
                    TrainingResult.Add(metricName, new List<double>());
                }

                double lossValue = totalBatchLossList.Average();
                double metricValue = totalMetricValueList.Average();
                TrainingResult["loss"].Add(lossValue);
                TrainingResult[metricName].Add(metricValue);
                metricsList.Add(metricName, metricValue);
                if (test != null)
                {
                    if (!TrainingResult.ContainsKey("val_loss"))
                    {
                        TrainingResult.Add("val_loss", new List<double>());
                    }

                    if (!TrainingResult.ContainsKey("val_" + metricName))
                    {
                        TrainingResult.Add("val_" + metricName, new List<double>());
                    }

                    int evalMiniBatchCount = 1;
                    List<double> totalEvalBatchLossList = new List<double>();
                    List<double> totalEvalMetricValueList = new List<double>();
                    while (test.NextBatch(evalMiniBatchCount, batchSize))
                    {
                        Variable actualVariable = CNTKLib.InputVariable(labelVariable.Shape, DataType.Float);
                        var evalLossFunc = Losses.Get(lossName, labelVariable, actualVariable);
                        var evalMetricFunc = Metrics.Get(metricName, labelVariable, actualVariable);
                        Value actual = EvaluateInternal(test.XFrame);
                        Value expected = GetValueBatch(test.YFrame);
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

                        evalMiniBatchCount++;
                    }

                    TrainingResult["val_loss"].Add(totalEvalBatchLossList.Average());
                    metricsList.Add("val_loss", totalEvalBatchLossList.Average());
                    TrainingResult["val_" + metricName].Add(totalEvalMetricValueList.Average());
                    metricsList.Add("val_" + metricName, totalEvalMetricValueList.Average());
                }

                OnEpochEnd(currentEpoch, trainer.TotalNumberOfSamplesSeen(), lossValue, metricsList);
                currentEpoch++;
            }

            OnTrainingEnd(TrainingResult);
        }

        private Value EvaluateInternal(DataFrame data)
        {
            Value features = GetValueBatch(data);
            var inputDataMap = new Dictionary<Variable, Value>() { { featureVariable, features } };
            var outputDataMap = new Dictionary<Variable, Value>() { { modelOut.Output, null } };
            modelOut.Evaluate(inputDataMap, outputDataMap, GlobalParameters.Device);
            return outputDataMap[modelOut.Output];
        }

        public IList<float> Evaluate(DataFrame data)
        {
            var outputValue = EvaluateInternal(data);
            IList<IList<float>> resultSet = outputValue.GetDenseData<float>(modelOut.Output);
            var result = resultSet.Select(x => x.First()).ToList();
            return result;
        }

        private Value GetValueBatch(DataFrame frame)
        {
            DataTable dt = frame.Frame.ToTable();
            int dim = dt.Columns.Count;
            List<float> batch = new List<float>();
            foreach (DataRow item in dt.Rows)
            {
                foreach (var row in item.ItemArray)
                {
                    if (row != null)
                    {
                        batch.Add((float)row);
                    }
                    else
                    {
                        batch.Add(0);
                    }
                }
            }

            Value result = Value.CreateBatch(new int[] { dim }, batch, GlobalParameters.Device);
            return result;
        }
    }
}
