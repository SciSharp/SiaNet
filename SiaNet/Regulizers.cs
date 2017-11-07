using CNTK;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SiaNet
{
    /// <summary>
    /// Regularizers allow to apply penalties on layer parameters or layer activity during optimization. These penalties are incorporated in the loss function that the network optimizes
    /// </summary>
    public class Regulizers
    {
        /// <summary>
        /// Gets or sets the l1 weight.
        /// </summary>
        /// <value>The l1.</value>
        public double L1 { get; set; }

        /// <summary>
        /// Gets or sets the l2 weight.
        /// </summary>
        /// <value>The l2.</value>
        public double L2 { get; set; }

        public bool IsL1 { get; set; }

        public bool IsL2 { get; set; }

        /// <summary>
        /// Initialize with l1 and l2 weight
        /// </summary>
        /// <param name="l1">The l1 weight value.</param>
        /// <param name="l2">The l2 weight value.</param>
        /// <returns>Regulizers.</returns>
        public static Regulizers L1L2(double l1 =0.01, double l2 =0.01)
        {
            Regulizers result = new Regulizers()
            {
                IsL1 = true,
                IsL2 = true,
                L1 = l1,
                L2 = l2
            };

            return result;
        }

        /// <summary>
        /// nitialize with l1 weight
        /// </summary>
        /// <param name="l1">The l1 weight value.</param>
        /// <returns>Regulizers.</returns>
        public static Regulizers RegL1(double l1 = 0.01)
        {
            Regulizers result = new Regulizers()
            {
                IsL1 = true,
                L1 = l1,
            };

            return result;
        }

        /// <summary>
        /// nitialize with l2 weight
        /// </summary>
        /// <param name="l2">The l2 weight value.</param>
        /// <returns>Regulizers.</returns>
        public static Regulizers RegL2(double l2 = 0.01)
        {
            Regulizers result = new Regulizers()
            {
                IsL2 = true,
                L2 = l2,
            };

            return result;
        }
    }
}
