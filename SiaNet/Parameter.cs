namespace SiaNet
{
    using SiaNet.Constraints;
    using SiaNet.Regularizers;
    using SiaNet.Engine;

    /// <summary>
    /// Placeholder variable for holding weight and bias for the neural network. Attached with constraints and regularizer to easy apply them during optimizer update operations
    /// </summary>
    public class Parameter : BaseParameter
    {
        /// <summary>
        /// The constraint
        /// </summary>
        private BaseConstraint constraint;

        /// <summary>
        /// The regularizer
        /// </summary>
        private BaseRegularizer regularizer;

        /// <summary>
        /// Initializes a new instance of the <see cref="Parameter"/> class.
        /// </summary>
        /// <param name="name">The name of the parameter.</param>
        /// <param name="shape">The shape of the weight/bias parameter.</param>
        public Parameter(string name, params long[] shape)
            :base (name, shape)
        {
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="Parameter"/> class.
        /// </summary>
        /// <param name="name">The name of the parameter.</param>
        /// <param name="dataType">Data type</param>
        /// <param name="shape">The shape of weight/bias parameter.</param>
        public Parameter(string name, DataType dataType, params long[] shape)
            : base(name, dataType, shape)
        {
        }

        /// <summary>
        /// Create an instance of parameter with tensor data.
        /// </summary>
        /// <param name="data">The tensor data to build the parameter.</param>
        /// <param name="name">The name of the parameter.</param>
        /// <returns></returns>
        public static Parameter Create(Tensor data, string name = "")
        {
            if (string.IsNullOrWhiteSpace(name))
                name = "v";

            Parameter x = new Parameter(name, data.ElementType, data.Shape);
            x.Data = data;

            return x;
        }

        /// <summary>
        /// Gets a value indicating whether the parameter have regularizer function attached.
        /// </summary>
        /// <value>
        ///   <c>true</c> if [have regularizer]; otherwise, <c>false</c>.
        /// </value>
        public bool HaveRegularizer
        {
            get
            {
                return regularizer != null;
            }
        }

        /// <summary>
        /// Sets the constraint function.
        /// </summary>
        /// <param name="fn">The function.</param>
        public void SetConstraint(BaseConstraint fn)
        {
            constraint = fn;
        }

        /// <summary>
        /// Sets the regularizer function.
        /// </summary>
        /// <param name="fn">The function.</param>
        public void SetRegularizer(BaseRegularizer fn)
        {
            regularizer = fn;
        }

        /// <summary>
        /// Applies the constraint function to weight/bias during training.
        /// </summary>
        public void ApplyConstraint()
        {
            if (constraint != null)
            {
                Data = constraint.Call(Data);
            }
        }

        /// <summary>
        /// Applies the regularizer function to weight/bias during training.
        /// </summary>
        /// <returns></returns>
        public float ApplyRegularizer()
        {
            float r = 0;
            if (regularizer != null)
            {
                r = regularizer.Call(Data);
            }

            return r;
        }

        /// <summary>
        /// Applies the gradient of regularizer function during back propagation.
        /// </summary>
        public void ApplyDeltaRegularizer()
        {
            if (regularizer != null)
            {
                Grad += regularizer.CalcGrad(Data);
            }
        }
    }
}
