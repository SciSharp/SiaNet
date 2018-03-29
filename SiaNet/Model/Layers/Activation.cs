using Newtonsoft.Json;

namespace SiaNet.Model.Layers
{
    /// <summary>
    ///     Applies an activation function to an output.
    /// </summary>
    /// <seealso cref="SiaNet.Model.LayerConfig" />
    public class Activation : LayerConfig
    {
        /// <summary>
        ///     Initializes a new instance of the <see cref="Activation" /> class.
        /// </summary>
        /// <param name="act">The name of activation function to use. <see cref="SiaNet.Common.OptActivations" /></param>
        public Activation(string act)
            : this()
        {
            Act = act;
        }

        /// <summary>
        ///     Initializes a new instance of the <see cref="Activation" /> class.
        /// </summary>
        internal Activation()
        {
            Name = "Activation";
        }

        /// <summary>
        ///     The name of activation function to use.
        /// </summary>
        /// <value>
        ///     The activation function name.
        /// </value>
        [JsonIgnore]
        public string Act
        {
            get => GetParam<string>("Act");

            set => SetParam("Act", value);
        }
    }
}