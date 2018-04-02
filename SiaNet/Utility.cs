namespace SiaNet
{
    using CNTK;
    using SiaNet.Model.Initializers;
    using System;

    internal class Utility
    {
        internal static Variable CreateParamVar(float value)
        {
            return new Parameter(new int[] { 1 }, DataType.Float, value);
        }

        internal static Initializer GetInitializerFromObject(object input, Initializer defaultValue)
        {
            Initializer result = defaultValue;

            if (input !=null)
            {
                if (input != null)
                {
                    if (input.GetType().Name.ToUpper() == "STRING")
                    {
                        result = new Initializer(input.ToString());
                    }
                    else if(input.GetType().BaseType.Name.ToUpper() == "INITIALIZER")
                    {
                        result = (Initializer)input;
                    }
                    else
                    {
                        throw new Exception("Only string or valid initializer instance allowed");
                    }
                }
            }

            return result;
        }
    }
}
