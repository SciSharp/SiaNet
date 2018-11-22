using CNTK;

namespace SiaNet.Data
{
    public class Constant : Variable
    {
        protected CNTK.Constant UnderlyingConstant;

        internal Constant(CNTK.Constant constant) : base(constant)
        {
            UnderlyingConstant = constant;
        }

        public static implicit operator CNTK.Constant(Constant v)
        {
            return v.UnderlyingConstant;
        }

        public static implicit operator Constant(CNTK.Constant v)
        {
            return new Constant(v);
        }

        public static implicit operator Constant(float f)
        {
            return CNTK.Constant.Scalar(DataType.Float, f, GlobalParameters.Device);
        }
    }
}