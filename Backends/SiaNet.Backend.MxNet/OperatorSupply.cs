using mx_float = System.Single;

// ReSharper disable once CheckNamespace
namespace SiaNet.Backend.MxNetLib
{
    public static class OperatorSupply
    {

        public static Symbol Plus(Symbol lhs, Symbol rhs)
        {
            return new Operator("_Plus").Set(lhs, rhs).CreateSymbol();
        }

        public static Symbol Sum(this Symbol lhs)
        {
            return new Operator("_Plus").Set(lhs).CreateSymbol();
        }

        public static Symbol Mul(Symbol lhs, Symbol rhs)
        {
            return new Operator("_Mul").Set(lhs, rhs).CreateSymbol();
        }

        public static Symbol Minus(Symbol lhs, Symbol rhs)
        {
            return new Operator("_Minus").Set(lhs, rhs).CreateSymbol();
        }

        public static Symbol Div(Symbol lhs, Symbol rhs)
        {
            return new Operator("_Div").Set(lhs, rhs).CreateSymbol();
        }

        public static Symbol Mod(Symbol lhs, Symbol rhs)
        {
            return new Operator("_Mod").Set(lhs, rhs).CreateSymbol();
        }

        public static Symbol Power(Symbol lhs, Symbol rhs)
        {
            return new Operator("_Power").Set(lhs, rhs).CreateSymbol();
        }

        public static Symbol Maximum(Symbol lhs, Symbol rhs)
        {
            return new Operator("_Maximum").Set(lhs, rhs).CreateSymbol();
        }

        public static Symbol Minimum(Symbol lhs, Symbol rhs)
        {
            return new Operator("_Minimum").Set(lhs, rhs).CreateSymbol();
        }

        public static Symbol Log(Symbol data)
        {
            return new Operator("log").SetInput("data", data).CreateSymbol();
        }

        public static Symbol PlusScalar(Symbol lhs, mx_float scalar)
        {
            return new Operator("_PlusScalar").Set(lhs)
                     .SetParam("scalar", scalar)
                     .CreateSymbol();
        }

        public static Symbol MinusScalar(Symbol lhs, mx_float scalar)
        {
            return new Operator("_MinusScalar").Set(lhs)
                     .SetParam("scalar", scalar)
                     .CreateSymbol();
        }

        public static Symbol RMinusScalar(mx_float scalar, Symbol rhs)
        {
            return new Operator("_RMinusScalar").Set(rhs)
                     .SetParam("scalar", scalar)
                     .CreateSymbol();
        }

        public static Symbol MulScalar(Symbol lhs, mx_float scalar)
        {
            return new Operator("_MulScalar").Set(lhs)
                     .SetParam("scalar", scalar)
                     .CreateSymbol();
        }

        public static Symbol DivScalar(Symbol lhs, mx_float scalar)
        {
            return new Operator("_DivScalar").Set(lhs)
                     .SetParam("scalar", scalar)
                     .CreateSymbol();
        }

        public static Symbol RDivScalar(mx_float scalar, Symbol rhs)
        {
            return new Operator("_RDivScalar").Set(rhs)
                     .SetParam("scalar", scalar)
                     .CreateSymbol();
        }

        public static Symbol ModScalar(Symbol lhs, mx_float scalar)
        {
            return new Operator("_ModScalar").Set(lhs)
                     .SetParam("scalar", scalar)
                     .CreateSymbol();
        }

        public static Symbol RModScalar(mx_float scalar, Symbol rhs)
        {
            return new Operator("_RModScalar").Set(rhs)
                     .SetParam("scalar", scalar)
                     .CreateSymbol();
        }

        public static Symbol PowerScalar(Symbol lhs, mx_float scalar)
        {
            return new Operator("_PowerScalar").Set(lhs)
                     .SetParam("scalar", scalar)
                     .CreateSymbol();
        }

        public static Symbol RPowerScalar(mx_float scalar, Symbol rhs)
        {
            return new Operator("_RPowerScalar").Set(rhs)
                     .SetParam("scalar", scalar)
                     .CreateSymbol();
        }

        public static Symbol MaximumScalar(Symbol lhs, mx_float scalar)
        {
            return new Operator("_MaximumScalar").Set(lhs)
                     .SetParam("scalar", scalar)
                     .CreateSymbol();
        }

        public static Symbol MinimumScalar(Symbol lhs, mx_float scalar)
        {
            return new Operator("_MinimumScalar").Set(lhs)
                     .SetParam("scalar", scalar)
                     .CreateSymbol();
        }

        public static Symbol Abs(Symbol data)
        {
            return new Operator("abs").Set(data)
                     .CreateSymbol();
        }

        public static Symbol Clip(Symbol data, float min, float max)
        {
            return new Operator("clip").Set(data)
                     .SetParam("a_min", min)
                     .SetParam("a_max", max)
                     .CreateSymbol();
        }

        public static Symbol Mean(Symbol data, Shape axis = null, bool keepdims = false, bool exclude = false)
        {
            return new Operator("mean")
            .SetParam("axis", axis)
            .SetParam("keepdims", keepdims)
            .SetParam("exclude", exclude)
            .SetInput("data", data)
            .CreateSymbol();
        }

        public static Symbol Square(Symbol data)
        {
            return new Operator("square")
            .SetInput("data", data)
            .CreateSymbol();
        }
    }
}
