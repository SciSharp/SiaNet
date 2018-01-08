namespace SiaNet
{
    using System;
    using System.Collections.Generic;
    using Newtonsoft.Json;
    using System.IO;
    using CNTK;
    using SiaNet.Model.Layers;
    using SiaNet.Processing;
    using SiaNet.Interface;
    using SiaNet.Common;
    using SiaNet.Model.Optimizers;
    using SiaNet.Model;
    using SiaNet.Events;

    /// <summary>
    /// The Sequential model is a linear stack of layers.
    /// </summary>
    /// <seealso cref="SiaNet.Model.ConfigModule" />
    public class Sequential : ConfigModule
    {
        /// <summary>
        /// Occurs when [on training start].
        /// </summary>
        public event On_Training_Start OnTrainingStart;

        /// <summary>
        /// Occurs when [on training end].
        /// </summary>
        public event On_Training_End OnTrainingEnd;

        /// <summary>
        /// Occurs when [on epoch start].
        /// </summary>
        public event On_Epoch_Start OnEpochStart;

        /// <summary>
        /// Occurs when [on epoch end].
        /// </summary>
        public event On_Epoch_End OnEpochEnd;

        /// <summary>
        /// Occurs when [on batch start].
        /// </summary>
        public event On_Batch_Start OnBatchStart;

        /// <summary>
        /// Occurs when [on batch end].
        /// </summary>
        public event On_Batch_End OnBatchEnd;

        int layerCounter = 1;

        public bool StopTraining = false;

        /// <summary>
        /// Gets the module.
        /// </summary>
        /// <value>The module.</value>
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

        private ITrainPredict trainPredict;

        private bool customBuilt = false;

        /// <summary>
        /// Gets or sets the training result.
        /// </summary>
        /// <value>The training result.</value>
        public Dictionary<string, List<double>> TrainingResult { get; set; }

        /// <summary>
        /// Gets or sets the stacked layers cofiguration.
        /// </summary>
        /// <value>The layers.</value>
        public List<LayerConfig> Layers { get; set; }

        /// <summary>
        /// Initializes a new instance of the <see cref="Sequential"/> class.
        /// </summary>
        public Sequential()
        {
            OnTrainingStart += Sequential_OnTrainingStart;
            OnTrainingEnd += Sequential_OnTrainingEnd;
            OnEpochStart += Sequential_OnEpochStart;
            OnEpochEnd += Sequential_OnEpochEnd;
            OnBatchStart += Sequential_OnBatchStart;
            OnBatchEnd += Sequential_OnBatchEnd;
            TrainingResult = new Dictionary<string, List<double>>();
            Layers = new List<LayerConfig>();
            learners = new List<Learner>();
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="Sequential"/> class.
        /// </summary>
        /// <param name="model">Model build outside and pass to this instance for training</param>
        /// <param name="feature">Feature variable</param>
        /// <param name="label">Label variable</param>
        public Sequential(Function model, Variable feature, Variable label)
            : this()
        {
            modelOut = model;
            featureVariable = feature;
            labelVariable = label;
            customBuilt = true;
        }

        /// <summary>
        /// Sequentials the on batch end.
        /// </summary>
        /// <param name="epoch">The epoch.</param>
        /// <param name="batchNumber">The batch number.</param>
        /// <param name="samplesSeen">The no. of samples seen.</param>
        /// <param name="loss">The loss.</param>
        /// <param name="metrics">The metrics.</param>
        private void Sequential_OnBatchEnd(int epoch, int batchNumber, uint samplesSeen, double loss, Dictionary<string, double> metrics)
        {
            if(StopTraining)
            {
                Environment.Exit(-1);
            }
        }

        /// <summary>
        /// Sequentials the on batch start.
        /// </summary>
        /// <param name="epoch">The epoch.</param>
        /// <param name="batchNumber">The batch number.</param>
        private void Sequential_OnBatchStart(int epoch, int batchNumber)
        {
            
        }

        /// <summary>
        /// Event triggered on every training epoch end
        /// </summary>
        /// <param name="epoch">The epoch.</param>
        /// <param name="samplesSeen">The no. samples seen.</param>
        /// <param name="loss">The loss.</param>
        /// <param name="metrics">The list of metrics.</param>
        private void Sequential_OnEpochEnd(int epoch, uint samplesSeen, double loss, Dictionary<string, double> metrics)
        {
            if (StopTraining)
            {
                Environment.Exit(-1);
            }
        }

        /// <summary>
        /// Event triggered on every training epoch start
        /// </summary>
        /// <param name="epoch">The epoch.</param>
        private void Sequential_OnEpochStart(int epoch)
        {

        }

        /// <summary>
        /// Event triggered on the complete training end
        /// </summary>
        /// <param name="trainingResult">The training result.</param>
        private void Sequential_OnTrainingEnd(Dictionary<string, List<double>> trainingResult)
        {

        }

        /// <summary>
        /// Event triggered on the training start
        /// </summary>
        private void Sequential_OnTrainingStart()
        {

        }

        /// <summary>
        /// Loads the neural network configuration saved using SaveNetConfig method.
        /// </summary>
        /// <param name="filepath">The filepath.</param>
        /// <returns>Sequential model</returns>
        public static Sequential LoadNetConfig(string filepath)
        {
            string json = File.ReadAllText(filepath);
            var result = JsonConvert.DeserializeObject<Sequential>(json);

            return result;
        }

        /// <summary>
        /// Saves the neural network configuration as json file.
        /// </summary>
        /// <param name="filepath">The filepath.</param>
        public void SaveNetConfig(string filepath)
        {
            string json = JsonConvert.SerializeObject(this, Formatting.Indented);
            File.WriteAllText(filepath, json);
        }

        /// <summary>
        /// Saves the complete model with training.
        /// </summary>
        /// <param name="filepath">The filepath.</param>
        public void SaveModel(string filepath)
        {
            modelOut.Save(filepath);
        }

        /// <summary>
        /// Loads the trained model for further prediction or training.
        /// </summary>
        /// <param name="filepath">The filepath.</param>
        public void LoadModel(string filepath)
        {
            modelOut = Function.Load(filepath, GlobalParameters.Device);
        }

        /// <summary>
        /// Stack the neural layers for building a deep learning model.
        /// </summary>
        /// <param name="config">The configuration.</param>
        public void Add(LayerConfig config)
        {
            if (customBuilt)
                throw new Exception("Cannot add layers to this sequential instance.");

            Layers.Add(config);
        }

        /// <summary>
        /// Configures the model for training.
        /// </summary>
        /// <param name="optimizer">The optimizer function name used for training the model.</param>
        /// <param name="loss">The function name with which the training loss will be minimized.</param>
        /// <param name="metric"> The metric name to be evaluated by the model during training and testing.</param>
        /// <param name="regulizer">The regulizer instance to apply penalty on layers parameters.</param>
        public void Compile(string optimizer, string loss, string metric, Regulizers regulizer = null)
        {
            CompileModel();
            BaseOptimizer optimizerInstance = new BaseOptimizer(optimizer);
            learners.Add(optimizerInstance.GetDefault(modelOut, regulizer));
            
            lossName = loss;
            lossFunc = Losses.Get(loss, labelVariable, modelOut);
            if (!string.IsNullOrWhiteSpace(metric))
            {
                metricName = metric;
                metricFunc = Metrics.Get(metric, labelVariable, modelOut);
            }
            else
            {
                metricName = loss;
                metricFunc = lossFunc;
            }
        }

        /// <summary>
        /// Configures the model for training.
        /// </summary>
        /// <param name="optimizer">The optimizer function name used for training the model.</param>
        /// <param name="loss">The function name with which the training loss will be minimized.</param>
        /// <param name="metric"> The metric name to be evaluated by the model during training and testing.</param>
        /// <param name="regulizer">The regulizer instance to apply penalty on layers parameters.</param>
        public void Compile(BaseOptimizer optimizer, string loss, string metric = "", Regulizers regulizer = null)
        {
            CompileModel();
            
            learners.Add(optimizer.Get(modelOut, regulizer));
            lossName = loss;
            lossFunc = Losses.Get(loss, labelVariable, modelOut);
            if (!string.IsNullOrWhiteSpace(metric))
            {
                metricName = metric;
                metricFunc = Metrics.Get(metric, labelVariable, modelOut);
            }
            else
            {
                metricName = loss;
                metricFunc = lossFunc;
            }
        }

        private void CompileModel()
        {
            if (customBuilt)
                return;

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

        /// <summary>
        /// Builds the stacked layer.
        /// </summary>
        /// <param name="layer">The layer.</param>
        /// <exception cref="System.InvalidOperationException"></exception>
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
                case OptLayers.LSTM:
                    var l14 = (LSTM)layer;
                    modelOut = NN.Recurrent.LSTM(modelOut, l14.Dim, l14.CellDim, l14.Activation, l14.RecurrentActivation, l14.WeightInitializer, l14.RecurrentInitializer, l14.UseBias, l14.BiasInitializer, l14.ReturnSequence);
                    break;
                case OptLayers.Reshape:
                    var l15 = (Reshape)layer;
                    modelOut = NN.Basic.Reshape(modelOut, l15.TargetShape);
                    break;
                default:
                    throw new InvalidOperationException(string.Format("{0} layer is not implemented."));
            }
        }

        /// <summary>
        /// Builds the first layer.
        /// </summary>
        /// <param name="layer">The layer.</param>
        /// <exception cref="System.ArgumentNullException">
        /// Input shape is missing for first layer
        /// </exception>
        /// <exception cref="System.InvalidOperationException"></exception>
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
                    featureVariable = Variable.InputVariable(new int[] { l2.Shape.Value }, DataType.Float);
                    modelOut = NN.Basic.BatchNorm(featureVariable, l2.Epsilon, l2.BetaInitializer, l2.GammaInitializer, l2.RunningMeanInitializer, l2.RunningStdInvInitializer, l2.Spatial, l2.NormalizationTimeConstant, l2.BlendTimeConst);
                    break;
                case OptLayers.Conv1D:
                    isConvolution = true;
                    var l3 = (Conv1D)layer;
                    if (l3.Shape == null)
                        throw new ArgumentNullException("Input shape is missing for first layer");
                    featureVariable = Variable.InputVariable(new int[] { l3.Shape.Item1, l3.Shape.Item2 }, DataType.Float);
                    modelOut = NN.Convolution.Conv1D(featureVariable, l3.Channels, l3.KernalSize, l3.Strides, l3.Padding, l3.Dialation, l3.Act, l3.UseBias, l3.WeightInitializer, l3.BiasInitializer);
                    break;
                case OptLayers.Conv2D:
                    isConvolution = true;
                    var l4 = (Conv2D)layer;
                    if (l4.Shape == null)
                        throw new ArgumentNullException("Input shape is missing for first layer");
                    featureVariable = Variable.InputVariable(new int[] { l4.Shape.Item1, l4.Shape.Item2, l4.Shape.Item3 }, DataType.Float);
                    modelOut = NN.Convolution.Conv2D(featureVariable, l4.Channels, l4.KernalSize, l4.Strides, l4.Padding, l4.Dialation, l4.Act, l4.UseBias, l4.WeightInitializer, l4.BiasInitializer);
                    break;
                case OptLayers.Conv3D:
                    isConvolution = true;
                    var l5 = (Conv3D)layer;
                    if (l5.Shape == null)
                        throw new ArgumentNullException("Input shape is missing for first layer");
                    featureVariable = Variable.InputVariable(new int[] { l5.Shape.Item1, l5.Shape.Item2, l5.Shape.Item3, l5.Shape.Item4 }, DataType.Float);
                    modelOut = NN.Convolution.Conv3D(featureVariable, l5.Channels, l5.KernalSize, l5.Strides, l5.Padding, l5.Dialation, l5.Act, l5.UseBias, l5.WeightInitializer, l5.BiasInitializer);
                    break;
                case OptLayers.Embedding:
                    var l6 = (Embedding)layer;
                    featureVariable = Variable.InputVariable(new int[] { l6.Shape }, DataType.Float);
                    modelOut = NN.Basic.Embedding(featureVariable, l6.EmbeddingDim, l6.Initializers);
                    break;
                case OptLayers.LSTM:
                    var l7 = (LSTM)layer;
                    if (l7.Shape == null)
                        throw new ArgumentNullException("Input shape is missing for first layer");
                    featureVariable = Variable.InputVariable(l7.Shape, DataType.Float);
                    modelOut = NN.Recurrent.LSTM(featureVariable, l7.Dim, l7.CellDim, l7.Activation, l7.RecurrentActivation, l7.WeightInitializer, l7.RecurrentInitializer, l7.UseBias, l7.BiasInitializer, l7.ReturnSequence);
                    break;
                case OptLayers.Reshape:
                    var l8 = (Reshape)layer;
                    if (l8.Shape == null)
                        throw new ArgumentNullException("Input shape is missing for first layer");
                    featureVariable = Variable.InputVariable(l8.Shape, DataType.Float);
                    modelOut = NN.Basic.Reshape(featureVariable, l8.TargetShape);
                    break;
                default:
                    throw new InvalidOperationException(string.Format("{0} cannot be used as first layer."));
            }
        }

        /// <summary>
        /// Trains the model for a fixed number of epochs.
        /// </summary>
        /// <param name="train">The training dataset.</param>
        /// <param name="epoches">The no. of trainin epoches.</param>
        /// <param name="batchSize">Size of the batch for training.</param>
        /// <param name="validation">The validation dataset.</param>
        /// <param name="shuffle">Shuffle the dataset while training</param>
        public void Train(XYFrame train, int epoches, int batchSize, XYFrame validation = null, bool shuffle = false)
        {
            OnTrainingStart();
            trainPredict = new DataFrameTrainPredict(modelOut, lossFunc, lossName, metricFunc, metricName, learners, featureVariable, labelVariable);
            TrainingResult = trainPredict.Train(train, validation, epoches, batchSize, OnEpochStart, OnEpochEnd, OnBatchStart, OnBatchEnd, shuffle);
            OnTrainingEnd(TrainingResult);
        }

        /// <summary>
        /// Trains the model for a fixed number of epochs.
        /// </summary>
        /// <param name="train">The train image generator.</param>
        /// <param name="epoches">The no. of trainin epoches.</param>
        /// <param name="batchSize">Size of the batch for training.</param>
        /// <param name="validation">The validation image generator.</param>
        public void Train(ImageDataGenerator train, int epoches, int batchSize, ImageDataGenerator validation = null)
        {
            OnTrainingStart();
            trainPredict = new ImgGenTrainPredict(modelOut, lossFunc, lossName, metricFunc, metricName, learners, featureVariable, labelVariable);
            TrainingResult = trainPredict.Train(train, validation, epoches, batchSize, OnEpochStart, OnEpochEnd, OnBatchStart, OnBatchEnd);
            OnTrainingEnd(TrainingResult);
        }

        /// <summary>
        /// Evaluates the specified data.
        /// </summary>
        /// <param name="data">The data for valuation.</param>
        /// <returns>List of prediction values</returns>
        public IList<float> Evaluate(DataFrame data)
        {
            return trainPredict.Evaluate(data);
        }
    }
}
