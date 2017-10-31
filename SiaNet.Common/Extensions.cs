using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SiaNet.Common
{
    public static class Extensions
    {
        public static double Mean(this List<double> values)
        {
            return values.Average();
        }

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
