namespace SiaNet.Regularizers
{
    using SiaNet.Engine;

    /// <summary>
    /// Regularizers allow to apply penalties on layer parameters or layer activity during optimization. These penalties are incorporated in the loss function that the network optimizes.
    /// <para>The penalties are applied on a per-layer basis.The exact API will depend on the layer, but the layers Dense, Conv1D, Conv2D and Conv3D have a unified API.</para>
    /// </summary>
    public abstract class BaseRegularizer
    {
        internal IBackend K = Global.CurrentBackend;

        /// <summary>
        /// Gets or sets the name of the regularizer function
        /// </summary>
        /// <value>
        /// The name.
        /// </value>
        public string Name { get; set; }

        /// <summary>
        /// Gets or sets the l1 value.
        /// </summary>
        /// <value>
        /// The l1.
        /// </value>
        internal float L1 { get; set; }

        /// <summary>
        /// Gets or sets the l2 value.
        /// </summary>
        /// <value>
        /// The l2.
        /// </value>
        internal float L2 { get; set; }

        /// <summary>
        /// Initializes a new instance of the <see cref="BaseRegularizer"/> class.
        /// </summary>
        /// <param name="l1">The l1 value. Default to 0.01</param>
        /// <param name="l2">The l2 value. Default to 0.01</param>
        public BaseRegularizer(float l1 = 0.01f, float l2 = 0.01f)
        {
            L1 = l1;
            L2 = l2;
        }

        /// <summary>
        /// Invoke the regularizer function during the model training
        /// </summary>
        /// <param name="x">The tensor data x.</param>
        /// <returns></returns>
        internal abstract float Call(Tensor x);

        /// <summary>
        /// Calculates the gradient of the regularizer function.
        /// </summary>
        /// <param name="x">The tensor data x.</param>
        /// <returns></returns>
        internal abstract Tensor CalcGrad(Tensor x);
    }
}
