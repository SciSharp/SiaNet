using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SiaNet.Common
{
    /// <summary>
    /// Extention class for various extension methods for generic list
    /// </summary>
    public static class Extensions
    {
        /// <summary>
        /// Means the specified values.
        /// </summary>
        /// <param name="values">The values.</param>
        /// <returns>System.Double.</returns>
        public static double Mean(this List<double> values)
        {
            return values.Average();
        }

        /// <summary>
        /// Standards the specified values.
        /// </summary>
        /// <param name="values">The values.</param>
        /// <returns>System.Double.</returns>
        public static double Std(this List<double> values)
        {
            double average = values.Average();
            double sumOfDerivation = 0;
            foreach (double value in values)
            {
                sumOfDerivation += (value) * (value);
            }

            double sumOfDerivationAverage = sumOfDerivation / (values.Count - 1);
            return Math.Sqrt(sumOfDerivationAverage - (average * average));
        }
    }
}
