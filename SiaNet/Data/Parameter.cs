using CNTK;

namespace SiaNet.Data
{
    public class Parameter : Variable
    {
        protected CNTK.Parameter UnderlyingParameter;

        internal Parameter(CNTK.Parameter constant) : base(constant)
        {
            UnderlyingParameter = constant;
        }

        public static implicit operator CNTK.Parameter(Parameter v)
        {
            return v.UnderlyingParameter;
        }

        public static implicit operator Parameter(CNTK.Parameter v)
        {
            return new Parameter(v);
        }

        public static implicit operator Parameter(float f)
        {
            return new CNTK.Parameter((Shape) 1, DataType.Float, f);
        }
    }
}