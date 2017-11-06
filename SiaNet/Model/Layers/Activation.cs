namespace SiaNet.Model.Layers
{
    using System.Dynamic;

    /// <summary>
    /// Applies an activation function to an output.
    /// </summary>
    /// <seealso cref="SiaNet.Model.LayerConfig" />
    public class Activation : LayerConfig
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="Activation"/> class.
        /// </summary>
        public Activation()
        {
            base.Name = "Activation";
            base.Params = new ExpandoObject();
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="Activation"/> class.
        /// </summary>
        /// <param name="act">The name of activation function to use. <see cref="SiaNet.Common.OptActivations"/></param>
        public Activation(string act)
            : this()
        {
            Act = act;
        }

        /// <summary>
        /// The name of activation function to use.
        /// </summary>
        /// <value>
        /// The activation function name.
        /// </value>
        [Newtonsoft.Json.JsonIgnore]
        public string Act
        {
            get
            {
                return base.Params.Act;
            }

            set
            {
                base.Params.Act = value;
            }
        }
    }
}
