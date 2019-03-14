using SiaNet.Engine;
using System;
using System.Collections.Generic;
using System.Text;

namespace SiaNet.Constraints
{
    /// <summary>
    /// MaxNorm weight constraint. Constrains the weights incident to each hidden unit to have a norm less than or equal to a desired value.
    /// </summary>
    public class MaxNorm : BaseConstraint
    {
        /// <summary>
        /// The maximum norm for the incoming weights.
        /// </summary>
        public float MaxValue { get; set; }

        /// <summary>
        /// Integer, axis along which to calculate weight norms. 
        /// For instance, in a Dense layer the weight matrix has shape (input_dim, output_dim), set axis to 0 to constrain each weight vector of length (input_dim,). 
        /// In a Conv2D layer, the weight tensor has shape  (input_depth, output_depth, rows, cols), set axis to [1, 2, 3] to constrain the weights of each filter tensor of size  (input_depth, rows, cols).
        /// </summary>
        public int Axis { get; set; }

        /// <summary>Initializes a new instance of the <see cref="MaxNorm"/> class.</summary>
        /// <param name="maxValue">The maximum norm for the incoming weights.</param>
        /// <param name="axis">Integer, axis along which to calculate weight norms. </param>
        public MaxNorm(float maxValue = 2, int axis = 0)
        {
            MaxValue = maxValue;
            Axis = axis;
        }

        /// <summary>Invokes the constraints</summary>
        /// <param name="w">The weight tensor</param>
        /// <returns></returns>
        internal override Tensor Call(Tensor w)
        {
            Tensor norms = K.Sqrt(K.Sum(K.Square(w), Axis));

            var desired = K.Clip(norms, 0, MaxValue);
            return w * (desired / (K.Epsilon() + norms));
        }
    }
}
