namespace SiaNet
{
    using System.Collections.Generic;
    using SiaNet.Layers;
    using SiaNet.Losses;
    using SiaNet.Metrics;
    using SiaNet.Optimizers;
    using System.IO;
    using Newtonsoft.Json;
    using SiaNet.Engine;

    /// <summary>
    /// The Sequential model is a linear stack of layers. Use model to add layers with Add method. Train function to train the layers with dataset. Predict function to invoke prediction against new data.
    /// </summary>
    public partial class Sequential
    {
        /// <summary>
        /// Stach of layers for this model
        /// </summary>
        /// <value>
        /// The layers.
        /// </value>
        public List<BaseLayer> Layers { get; set; }

        /// <summary>
        /// Gets or sets the loss function.
        /// </summary>
        /// <value>
        /// The loss function.
        /// </value>
        internal BaseLoss LossFn { get; set; }

        /// <summary>
        /// Gets or sets the metric function.
        /// </summary>
        /// <value>
        /// The metric function.
        /// </value>
        internal BaseMetric MetricFn { get; set; }

        /// <summary>
        /// Gets or sets the optimizer function.
        /// </summary>
        /// <value>
        /// The optimizer function.
        /// </value>
        internal BaseOptimizer OptimizerFn { get; set; }

        /// <summary>
        /// Initializes a new instance of the <see cref="Sequential"/> class.
        /// </summary>
        public Sequential()
        {
            Layers = new List<BaseLayer>();
        }

        /// <summary>
        /// Function to add layer to the sequential model.
        /// <para>
        /// Example use:
        /// <code>
        /// var model = new Sequential();
        /// model.EpochEnd += Model_EpochEnd;
        /// model.Add(new Dense(100, ActType.ReLU));
        /// model.Add(new Dense(50, ActType.ReLU));
        /// model.Add(new Dense(1, ActType.Sigmoid));
        /// </code>
        /// </para>
        /// </summary>
        /// <param name="l">The l.</param>
        public void Add(BaseLayer l)
        {
            Layers.Add(l);
        }

        /// <summary>
        /// Forwards the specified input.
        /// </summary>
        /// <param name="input">The input.</param>
        /// <returns></returns>
        private Tensor Forward(Tensor input)
        {
            BaseLayer lastLayer = null;

            foreach (var layer in Layers)
            {
                if (lastLayer == null)
                    layer.Forward(input);
                else
                    layer.Forward(lastLayer.Output);

                lastLayer = layer;
            }

            return lastLayer.Output;
        }

        /// <summary>
        /// Backwards the specified grad output.
        /// </summary>
        /// <param name="gradOutput">The grad output.</param>
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

        /// <summary>
        /// Applies the regularizer.
        /// </summary>
        /// <param name="loss">The loss.</param>
        /// <returns></returns>
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

        /// <summary>
        /// Applies the delta regularizer.
        /// </summary>
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

        /// <summary>
        /// Before training a model, you need to configure the learning process, which is done via the compile method. It receives three arguments:
        /// <para>
        /// 1. An optimizer.This could be the OptimizerType enum or an instance of the Optimizer class.
        /// </para><para>
        /// 2. A loss function.This is the objective that the model will try to minimize.
        /// It can be the LossType enum identifier of an existing loss function (such as CategoricalCrossentropy or MeanSquaredError), or it can be an instance of the loss class.
        /// </para>
        /// <para>
        /// 3. A metric function.For any classification problem you will want the Accuracy of model. A metric could be the MetricType enum or instance of metric class.
        /// </para>
        /// </summary>
        /// <param name="optimizer">The optimizer type.</param>
        /// <param name="loss">The loss type.</param>
        /// <param name="metric">The metric type.</param>
        public void Compile(OptimizerType optimizer, LossType loss, MetricType metric)
        {
            OptimizerFn = BaseOptimizer.Get(optimizer);
            LossFn = BaseLoss.Get(loss);
            MetricFn = BaseMetric.Get(metric);
        }

        /// <summary>
        /// Before training a model, you need to configure the learning process, which is done via the compile method. It receives three arguments:
        /// <para>
        /// 1. An optimizer.This could be the OptimizerType enum or an instance of the Optimizer class.
        /// </para><para>
        /// 2. A loss function.This is the objective that the model will try to minimize.
        /// It can be the LossType enum identifier of an existing loss function (such as CategoricalCrossentropy or MeanSquaredError), or it can be an instance of the loss class.
        /// </para>
        /// <para>
        /// 3. A metric function.For any classification problem you will want the Accuracy of model. A metric could be the MetricType enum or instance of metric class.
        /// </para>
        /// </summary>
        /// <param name="optimizer">The optimizer instance.</param>
        /// <param name="loss">The loss type.</param>
        /// <param name="metric">The metric type.</param>
        public void Compile(BaseOptimizer optimizer, LossType loss, MetricType metric)
        {
            OptimizerFn = optimizer;
            LossFn = BaseLoss.Get(loss);
            MetricFn = BaseMetric.Get(metric);
        }

        /// <summary>
        /// Saves the model in json format to the file path.
        /// </summary>
        /// <param name="filePath">The file path.</param>
        public void SaveModel(string filePath)
        {
            string modelJson = JsonConvert.SerializeObject(this
                                                    , Formatting.Indented
                                                    , new JsonSerializerSettings() { TypeNameHandling = TypeNameHandling.Auto });
            File.WriteAllText(filePath, modelJson);
        }

        /// <summary>
        /// Loads the model from the saved json to the Sequential model instance.
        /// </summary>
        /// <param name="filePath">The file path.</param>
        /// <returns></returns>
        public static Sequential LoadModel(string filePath)
        {
            string jsondata = File.ReadAllText(filePath);
            Sequential model = JsonConvert.DeserializeObject<Sequential>(jsondata, new JsonSerializerSettings() { TypeNameHandling = TypeNameHandling.Auto });
            return model;
        }
    }
}
