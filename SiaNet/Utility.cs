namespace SiaNet
{
    using CNTK;

    public class Utility
    {
        public static Variable CreateParamVar(float value)
        {
            return new Parameter(new int[] { 1 }, DataType.Float, value);
        }
    }
}
