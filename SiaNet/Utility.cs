using CNTK;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SiaNet
{
    public class Utility
    {
        public static Variable CreateParamVar(float value)
        {
            return new Parameter(new int[] { 1 }, DataType.Float, value);
        }
    }
}
