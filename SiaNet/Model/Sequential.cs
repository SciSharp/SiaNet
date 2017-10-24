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

        public static Sequential LoadConfig(string filepath, bool fromXml = false)
        {
            string json = File.ReadAllText(filepath);
            var result = JsonConvert.DeserializeObject<Sequential>(json);

            return result;
        }

        public void SaveConfig(string filepath, bool toXml = false)
        {
            string json = JsonConvert.SerializeObject(this, Formatting.Indented);
            File.WriteAllText(filepath, json);
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
                    modelOut = NN.Basic.BatchNorm(modelOut);
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
                    featureVariable = Variable.InputVariable(new int[] { l1.Shape.Value }, DataType.Float);
                    modelOut = NN.Basic.Dense(featureVariable, l1.Dim, l1.Act, l1.UseBias, l1.WeightInitializer, l1.BiasInitializer);
                    break;
                case OptLayers.BatchNorm:
                    var l2 = (BatchNorm)layer;
                    break;
                default:
                    throw new InvalidOperationException(string.Format("{0} cannot be used as first layer."));
            }
        }

        public void Train(XYFrame train, int batchSize, int epoches, XYFrame test = null)
        {
            OnTrainingStart();
            metricFunc.Save("metr.txt");
            var trainer = Trainer.CreateTrainer(modelOut, lossFunc, metricFunc, learners);
            int currentEpoch = 1;
            while (currentEpoch <= epoches)
            {
                OnEpochStart(currentEpoch);
                int miniBatchCount = 1;
                while (train.NextBatch(miniBatchCount, batchSize))
                {
                    Value features = GetValueBatch(train.CurrentBatch.XFrame);
                    Value labels = GetValueBatch(train.CurrentBatch.YFrame);

                    trainer.TrainMinibatch(new Dictionary<Variable, Value>() { { featureVariable, features }, { labelVariable, labels } }, GlobalParameters.Device);
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

                double lossValue = trainer.PreviousMinibatchLossAverage();
                double metricValue = trainer.PreviousMinibatchEvaluationAverage();
                TrainingResult["loss"].Add(lossValue);
                TrainingResult[metricName].Add(metricValue);

                if (test != null)
                {
                    Value actual = EvaluateInternal(test.XFrame);
                    Value expected = GetValueBatch(test.YFrame);
                    var inputDataMap = new Dictionary<Variable, Value>() { { labelVariable, actual } };
                    var outputDataMap = new Dictionary<Variable, Value>() { { labelVariable, expected } };

                    lossFunc.Evaluate(inputDataMap, outputDataMap, GlobalParameters.Device);
                    IList<IList<float>> resultSet = outputDataMap[labelVariable].GetDenseData<float>(labelVariable);
                    var result = resultSet.Select(x => x.First()).ToList();
                }

                OnEpochEnd(currentEpoch, trainer.TotalNumberOfSamplesSeen(), lossValue, new Dictionary<string, double>() { { metricName, metricValue } });
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
