using CNTK;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SiaNet
{
    public class Regulizers
    {
        public double L1 { get; set; }

        public double L2 { get; set; }

        public bool IsL1 { get; set; }

        public bool IsL2 { get; set; }

        public static Regulizers L1L2(double l1 =0.01f, double l2 =0.01f)
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

        public static Regulizers RegL1(double l1 = 0.01f)
        {
            Regulizers result = new Regulizers()
            {
                IsL1 = true,
                L1 = l1,
            };

            return result;
        }

        public static Regulizers RegL2(double l2 = 0.01f)
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
