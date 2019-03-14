namespace SiaNet.Initializers
{
    using SiaNet.Engine;

    /// <summary>
    /// Base class for the initializer. Initializes the tensor weights or bias with values based on the type of the initializer selected.
    /// </summary>
    public abstract class BaseInitializer
    {
        internal IBackend K = Global.CurrentBackend;

        /// <summary>
        /// Gets or sets the name of the initializer.
        /// </summary>
        /// <value>
        /// The name.
        /// </value>
        public string Name { get; set; }

        /// <summary>
        /// Initializes a new instance of the <see cref="BaseInitializer"/> class.
        /// </summary>
        /// <param name="name">The name of the initializer.</param>
        public BaseInitializer(string name)
        {
            Name = name;
        }

        /// <summary>
        /// Generates a tensor with specified shape.
        /// </summary>
        /// <param name="shape">The shape of the tensor.</param>
        /// <returns></returns>
        public abstract Tensor Generate(params long[] shape);
    }
}
