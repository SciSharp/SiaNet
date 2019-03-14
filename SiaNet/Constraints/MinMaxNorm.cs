using SiaNet.Engine;
using System;
using System.Collections.Generic;
using System.Text;

namespace SiaNet.Constraints
{
    /// <summary>
    /// MinMaxNorm weight constraint. Constrains the weights incident to each hidden unit to have the norm between a lower bound and an upper bound.
    /// </summary>
    /// <seealso cref="SiaNet.Constraints.BaseConstraint" />
    public class MinMaxNorm : BaseConstraint
    {
        /// <summary>
        /// The minimum norm for the incoming weights.
        /// </summary>
        /// <value>
        /// The minimum value.
        /// </value>
        public float MinValue { get; set; }

        /// <summary>
        /// The maximum norm for the incoming weights.
        /// </summary>
        /// <value>
        /// The maximum value.
        /// </value>
        public float MaxValue { get; set; }

        /// <summary>
        /// Rate for enforcing the constraint: weights will be rescaled to yield  (1 - rate) * norm + rate * norm.clip(min_value, max_value). 
        /// Effectively, this means that rate=1.0 stands for strict enforcement of the constraint, while rate&lt;1.0 means that weights will be rescaled at each step to slowly move towards a value inside the desired interval.
        /// </summary>
        /// <value>
        /// The rate.
        /// </value>
        public float Rate { get; set; }

        /// <summary>
        /// Integer, axis along which to calculate weight norms. For instance, in a Dense layer the weight matrix has shape (input_dim, output_dim), set axis to 0 to constrain each weight vector of length (input_dim,). 
        /// In a Conv2D layer, the weight tensor has shape  (output_depth, input_depth, rows, cols), 
        /// set axis to [1, 2, 3] to constrain the weights of each filter tensor of size  (input_depth, rows, cols).
        /// </summary>
        /// <value>
        /// The axis.
        /// </value>
        public uint Axis { get; set; }

        /// <summary>
        /// Initializes a new instance of the <see cref="MinMaxNorm"/> class.
        /// </summary>
        /// <param name="minVale">The minimum norm for the incoming weights.</param>
        /// <param name="maxValue">The maximum norm for the incoming weights.</param>
        /// <param name="rate">Rate for enforcing the constraint: weights will be rescaled to yield  (1 - rate) * norm + rate * norm.clip(min_value, max_value).</param>
        /// <param name="axis">Integer, axis along which to calculate weight norms. </param>
        public MinMaxNorm(float minVale = 0, float maxValue = 1, float rate = 1f, uint axis = 0)
        {
            MinValue = minVale;
            MaxValue = maxValue;
            Rate = rate;
            Axis = axis;
        }

        /// <summary>
        /// Invoke the constraint
        /// </summary>
        /// <param name="w">The weight tensor.</param>
        /// <returns></returns>
        internal override Tensor Call(Tensor w)
        {
            Tensor norms = null;
            norms = K.Sqrt(K.Sum(K.Square(w), (int)Axis));


            var desired = Rate * K.Clip(norms, MinValue, MaxValue) + (1 - Rate) * norms;
            w = w * (desired / (K.Epsilon() + norms));
            return w;
        }
    }
}
