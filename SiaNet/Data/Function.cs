namespace SiaNet.Data
{
    public class Function : Variable
    {
        protected CNTK.Function UnderlyingFunction;

        protected Function(CNTK.Function function) : base(function)
        {
            UnderlyingFunction = function;
        }

        public static implicit operator CNTK.Function(Function v)
        {
            return v.UnderlyingFunction;
        }

        public static implicit operator Function(CNTK.Function v)
        {
            return new Function(v);
        }
    }
}