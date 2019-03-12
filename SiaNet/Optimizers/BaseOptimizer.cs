namespace SiaNet.Optimizers
{
    using SiaNet.Engine;
    using SiaNet.Layers;

    /// <summary>
    /// 
    /// </summary>
    public abstract class BaseOptimizer 
    {
        internal IBackend K = Global.CurrentBackend;

        /// <summary>
        /// Gets or sets the name of the optimizer function
        /// </summary>
        /// <value>
        /// The name.
        /// </value>
        public string Name { get; set; }

        /// <summary>
        /// Gets or sets the learning rate for the optimizer.
        /// </summary>
        /// <value>
        /// The learning rate.
        /// </value>
        public float LearningRate { get; set; }

        /// <summary>
        /// Parameter that accelerates SGD in the relevant direction and dampens oscillations.
        /// </summary>
        /// <value>
        /// The momentum.
        /// </value>
        public float Momentum { get; set; }

        /// <summary>
        /// Learning rate decay over each update.
        /// </summary>
        /// <value>
        /// The decay rate.
        /// </value>
        public float DecayRate { get; set; }

        /// <summary>
        /// Initializes a new instance of the <see cref="BaseOptimizer"/> class.
        /// </summary>
        /// <param name="lr">The lr.</param>
        /// <param name="name">The name.</param>
        public BaseOptimizer(float lr, string name)
        {
            LearningRate = lr;
            Name = name;
        }

        /// <summary>
        /// Updates the specified iteration.
        /// </summary>
        /// <param name="iteration">The iteration.</param>
        /// <param name="layer">The layer.</param>
        internal abstract void Update(int iteration, BaseLayer layer);

        /// <summary>
        /// Gets the specified optimizer type.
        /// </summary>
        /// <param name="optimizerType">Type of the optimizer.</param>
        /// <returns></returns>
        internal static BaseOptimizer Get(OptimizerType optimizerType)
        {
            BaseOptimizer opt = null;
            switch (optimizerType)
            {
                case OptimizerType.SGD:
                    opt = new SGD();
                    break;
                case OptimizerType.Adamax:
                    opt = new Adamax();
                    break;
                case OptimizerType.RMSprop:
                    opt = new RMSProp();
                    break;
                case OptimizerType.Adagrad:
                    opt = new Adagrad();
                    break;
                case OptimizerType.Adadelta:
                    opt = new Adadelta();
                    break;
                case OptimizerType.Adam:
                    opt = new Adam();
                    break;
                default:
                    break;
            }

            return opt;
        }
    }
}
