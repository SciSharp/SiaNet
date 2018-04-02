using CNTK;

namespace SiaNet.Model
{
    public class Variable
    {
        protected CNTK.Variable UnderlyingVariable;

        internal Variable(CNTK.Variable variable)
        {
            UnderlyingVariable = variable;
        }
        

        public Shape Shape
        {
            get => UnderlyingVariable.Shape;
        }

        public static Function operator +(Variable left, Variable right)
        {
            return CNTKLib.Plus(left, right);
        }

        public static Function operator &(Variable left, Variable right)
        {
            return CNTKLib.ElementAnd(left, right);
        }

        public static Function operator |(Variable left, Variable right)
        {
            return CNTKLib.ElementOr(left, right);
        }

        public static Function operator /(Variable left, Variable right)
        {
            return CNTKLib.ElementDivide(left, right);
        }

        public static Function operator ==(Variable left, Variable right)
        {
            return CNTKLib.Equal(left, right);
        }

        public static Function operator ^(Variable left, Variable right)
        {
            return CNTKLib.ElementXor(left, right);
        }

        public static Function operator >(Variable left, Variable right)
        {
            return CNTKLib.Greater(left, right);
        }

        public static Function operator >=(Variable left, Variable right)
        {
            return CNTKLib.GreaterEqual(left, right);
        }

        public static implicit operator CNTK.Variable(Variable v)
        {
            return v.UnderlyingVariable;
        }

        public static implicit operator Variable(CNTK.Variable v)
        {
            return new Variable(v);
        }
        
        public static Function operator !=(Variable left, Variable right)
        {
            return CNTKLib.NotEqual(left, right);
        }

        public static Function operator <(Variable left, Variable right)
        {
            return CNTKLib.Less(left, right);
        }

        public static Function operator <=(Variable left, Variable right)
        {
            return CNTKLib.LessEqual(left, right);
        }

        public static Function operator !(Variable left)
        {
            return CNTKLib.Negate(left);
        }

        public static Function operator *(Variable left, Variable right)
        {
            return CNTKLib.ElementTimes(left, right);
        }

        public static Function operator -(Variable left, Variable right)
        {
            return CNTKLib.Minus(left, right);
        }

        public Function Abs()
        {
            return CNTKLib.Abs(this);
        }

        public Function Acos()
        {
            return CNTKLib.Acos(this);
        }

        public Function Asin()
        {
            return CNTKLib.Asin(this);
        }
        
        public Function ReduceMeanByAxes(int staticAxes)
        {
            return CNTKLib.ReduceMean(this, new Axis(staticAxes));
        }

        public Function ReduceSumByAxes(int staticAxes)
        {
            return CNTKLib.ReduceSum(this, new Axis(staticAxes));
        }

        public Function Square()
        {
            return CNTKLib.Square(this);
        }
        public Function Sqrt()
        {
            return CNTKLib.Sqrt(this);
        }
        public Function Clip(Variable min, Variable max)
        {
            return CNTKLib.Clip(this, min, max);
        }

        public Function Asinh()
        {
            return CNTKLib.Asinh(this);
        }

        public Function Atanh()
        {
            return CNTKLib.Atanh(this);
        }


        public Function Ceil()
        {
            return CNTKLib.Ceil(this);
        }

        public Function Flatten()
        {
            return CNTKLib.Flatten(this);
        }

        public Function Floor()
        {
            return CNTKLib.Floor(this);
        }

        public Function Log()
        {
            return CNTKLib.Log(this);
        }

        public Function Max(Variable variable, string variableName)
        {
            return CNTKLib.ElementMax(this, variable, variableName);
        }

        public Function Min(Variable variable, string variableName)
        {
            return CNTKLib.ElementMin(this, variable, variableName);
        }

        public Function Times(Variable variable, uint outputRank, int inferInputRankToMap)
        {
            return CNTKLib.Times(this, variable, outputRank,
                inferInputRankToMap);
        }

        public Function Times(Variable variable, uint outputRank)
        {
            return CNTKLib.Times(this, variable, outputRank);
        }

        public Function Times(Variable variable)
        {
            return CNTKLib.Times(this, variable);
        }
    }
}