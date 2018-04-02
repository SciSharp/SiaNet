using CNTK;
using Newtonsoft.Json;
using SiaNet.NN;

namespace SiaNet.Model.Layers
{
    /// <summary>
    ///     Reshapes an output to a certain shape.
    /// </summary>
    public class Reshape : OptimizableLayerBase
    {
        /// <summary>
        ///     Initializes a new instance of the <see cref="Reshape" /> class.
        /// </summary>
        /// <param name="targetshape">The target shape of the output.</param>
        public Reshape(int[] targetshape)
            : this()
        {
            TargetShape = targetshape;
        }

        /// <summary>
        ///     Initializes a new instance of the <see cref="Reshape" /> class.
        /// </summary>
        internal Reshape()
        {
        }

        /// <summary>
        ///     List of integers. Does not include the batch axis.
        /// </summary>
        /// <value>
        ///     The target shape.
        /// </value>
        [JsonIgnore]
        public int[] TargetShape
        {
            get => GetParam<int[]>("TargetShape");

            set => SetParam("TargetShape", value);
        }

        /// <inheritdoc />
        internal override Function ToFunction(Variable inputFunction)
        {
            return Basic.Reshape(inputFunction, TargetShape);
        }
    }
}