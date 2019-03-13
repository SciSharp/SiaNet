namespace SiaNet.Layers
{
    using SiaNet.Constraints;
    using SiaNet.Engine;
    using SiaNet.Initializers;
    using SiaNet.Regularizers;
    using System;
    using System.Collections.Generic;

    /// <summary>
    /// Base class for the layers
    /// </summary>
    public abstract class BaseLayer
    {
        /// <summary>
        /// The current backend
        /// </summary>
        internal IBackend K = Global.CurrentBackend;

        /// <summary>
        /// The parameters list used by the layer
        /// </summary>
        [NonSerialized]
        public Dictionary<string, Parameter> Params;

        /// <summary>
        /// The input tensor parameter
        /// </summary>
        [NonSerialized]
        public Parameter Input;

        /// <summary>
        /// The output tensor parameter
        /// </summary>
        [NonSerialized]
        public Tensor Output;

        /// <summary>
        /// Gets or sets the name of the layer
        /// </summary>
        /// <value>
        /// The name.
        /// </value>
        public string Name { get; set; }

        /// <summary>
        /// Gets or sets a value indicating whether the layer is train only
        /// </summary>
        /// <value>
        ///   <c>true</c> if [skip pred]; otherwise, <c>false</c>.
        /// </value>
        public bool SkipPred { get; set; }

        /// <summary>
        /// Initializes a new instance of the <see cref="BaseLayer"/> class.
        /// </summary>
        /// <param name="name">The name.</param>
        public BaseLayer(string name)
        {
            Name = K.UUID(name);
            Params = new Dictionary<string, Parameter>();
        }

        /// <summary>
        /// Forwards the inputs and compute the output
        /// </summary>
        /// <param name="x">The input tensor for this layer.</param>
        public virtual void Forward(Tensor x)
        {
            Input = x.ToParameter();
        }

        /// <summary>
        /// Calculate the gradient of this layer function
        /// </summary>
        /// <param name="outputgrad">The calculated output grad from previous layer.</param>
        public virtual void Backward(Tensor outputgrad)
        {

        }

        /// <summary>
        /// Gets or sets the <see cref="Parameter"/> with the specified name.
        /// </summary>
        /// <value>
        /// The <see cref="Parameter"/>.
        /// </value>
        /// <param name="name">The name.</param>
        /// <returns></returns>
        public Parameter this[string name]
        {
            get
            {
                return Params[Name + "_" +name];
            }
            set
            {
                Params[Name + "_" + name] = value;
            }
        }

        /// <summary>
        /// Builds the parameter with specified parameter. Used to create weight or bias parameter for the layer.
        /// </summary>
        /// <param name="name">The name of the parameter.</param>
        /// <param name="shape">The shape of the tensor.</param>
        /// <param name="elementType">Datatype of the tensor.</param>
        /// <param name="initializer">The initializer used to create the parameter.</param>
        /// <param name="constraint">The constraint function applied for this parameter.</param>
        /// <param name="regularizer">The regularizer function for this parameter.</param>
        /// <param name="trainable">if set to <c>true</c> [trainable].</param>
        /// <returns></returns>
        public Parameter BuildParam(string name, long[] shape, DataType elementType, BaseInitializer initializer, BaseConstraint constraint = null, BaseRegularizer regularizer = null, bool trainable = true)
        {
            Parameter v = null;
            name = Name + "_" + name;
            if (!Params.ContainsKey(name))
            {
                v = new Parameter(name, elementType, shape);
                v.Data = initializer.Generate(shape);
                v.SetConstraint(constraint);
                v.SetRegularizer(regularizer);
                if(trainable)
                    Params.Add(name, v);
            }
            else
            {
                v = Params[name];
            }

            return v;
        }
    }
}
