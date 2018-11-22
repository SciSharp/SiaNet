using System;
using CNTK;

namespace SiaNet.Data
{
    public class Variable : IEquatable<Variable>
    {
        protected readonly CNTK.Variable UnderlyingVariable;

        internal Variable(CNTK.Variable variable)
        {
            UnderlyingVariable = variable;
        }


        public Shape Shape
        {
            get => UnderlyingVariable.Shape;
        }

        /// <inheritdoc />
        public bool Equals(Variable other)
        {
            if (ReferenceEquals(null, other))
            {
                return false;
            }

            if (ReferenceEquals(this, other))
            {
                return true;
            }

            return UnderlyingVariable.Equals(other.UnderlyingVariable);
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

        /// <inheritdoc />
        public override bool Equals(object obj)
        {
            if (ReferenceEquals(null, obj))
            {
                return false;
            }

            if (ReferenceEquals(this, obj))
            {
                return true;
            }

            if (ReferenceEquals(null, obj as Variable))
            {
                return false;
            }

            return Equals((Variable) obj);
        }

        /// <inheritdoc />
        public override int GetHashCode()
        {
            return UnderlyingVariable != null ? UnderlyingVariable.GetHashCode() : 0;
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

        public Function Clip(Variable min, Variable max)
        {
            return CNTKLib.Clip(this, min, max);
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

        public Function ReduceMeanByAxes(int staticAxes)
        {
            return CNTKLib.ReduceMean(this, new Axis(staticAxes));
        }

        public Function ReduceSumByAxes(int staticAxes)
        {
            return CNTKLib.ReduceSum(this, new Axis(staticAxes));
        }

        public Function Sqrt()
        {
            return CNTKLib.Sqrt(this);
        }

        public Function Square()
        {
            return CNTKLib.Square(this);
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