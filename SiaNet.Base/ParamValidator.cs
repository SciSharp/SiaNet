using System;
using System.Collections.Generic;
using System.Text;
using System.Linq;

namespace SiaNet
{
    public class ParamValidator
    {
        public static void Validate(string name, string value, params string[] validValues)
        {
            if (!validValues.Contains(value))
            {
                string message = "Invalid value for " + name + ". Valid values are " + string.Join(", ", validValues);
                throw new Exception(message);
            }
        }
    }
}
