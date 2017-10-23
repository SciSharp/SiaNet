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

    public delegate void On_Training_End();

    public delegate void On_Epoch_Start(int epoch);

    public delegate void On_Epoch_End(int epoch, float loss, Dictionary<string, float> metrics);

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

        private string lossFunc;

        private string metricFunc;

        private Function modelOut;

        private bool isConvolution;

        private int[] shape;

        private int outputNums;

        public List<LayerConfig> Layers { get; set; }

        public Sequential()
        {
            OnTrainingStart += Sequential_OnTrainingStart;
            OnTrainingEnd += Sequential_OnTrainingEnd;
            OnEpochStart += Sequential_OnEpochStart;
            OnEpochEnd += Sequential_OnEpochEnd;
            Layers = new List<LayerConfig>();
            learners = new List<Learner>();
        }

        private void Sequential_OnEpochEnd(int epoch, float loss, Dictionary<string, float> metrics)
        {
            
        }

        private void Sequential_OnEpochStart(int epoch)
        {
            
        }

        private void Sequential_OnTrainingEnd()
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
            bool first = true;
            foreach (var item in Layers)
            {
                if(first)
                {
                    BuildFirstLayer(item);
                    first = false;
                    continue;
                }

                BuildStackedLayer(item);
            }

            outputNums = modelOut.Output.Shape[0];

            learners.Add(Optimizers.Get(optimizer, modelOut));
            lossFunc = loss;
            metricFunc = metric;
        }

        public void Compile(Learner optimizer, string loss, string metric)
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

            outputNums = modelOut.Output.Shape[0];

            learners.Add(optimizer);
            lossFunc = loss;
            metricFunc = metric;
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
                    modelOut = NN.Basic.Dense(l1.Shape.Value, l1.Dim, l1.Act, l1.UseBias, l1.WeightInitializer, l1.BiasInitializer);
                    shape = new int[] { l1.Shape.Value };
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
            Variable featureVariable = Variable.InputVariable(shape, DataType.Float);
            Variable labelVariable = Variable.InputVariable(new int[] { outputNums }, DataType.Float);
            Function loss = Losses.Get(lossFunc, new Variable(modelOut), labelVariable);
            Function metric = Metrics.Get(metricFunc, new Variable(modelOut), labelVariable);
            var trainer = Trainer.CreateTrainer(modelOut, loss, metric, learners);
            int currentEpoch = 1;
            while (currentEpoch <= epoches)
            {
                OnEpochStart(epoches);
                int miniBatchCount = 1;
                while(train.NextBatch(miniBatchCount, batchSize))
                {
                    Value features = GetValueBatch(train.XFrame);
                    Value labels = GetValueBatch(train.YFrame);

                    trainer.TrainMinibatch(new Dictionary<Variable, Value>() { { featureVariable, features }, { labelVariable, labels } }, GlobalParameters.Device);
                    miniBatchCount++;
                }

                Function trainedModel = trainer.Model();

                OnEpochEnd(currentEpoch, 0, null);
                currentEpoch++;
            }

            OnTrainingEnd();
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

            Value result = Value.CreateBatch<float>(new int[] { dim }, batch, GlobalParameters.Device);
            return result;
        }
    }
}
