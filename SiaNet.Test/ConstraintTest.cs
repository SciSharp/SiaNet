using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorSharp;

namespace SiaNet.Test
{
    [TestClass]
    public class ConstraintTest
    {
        [TestMethod]
        public void MaxNormTest()
        {
            Tensor x = Tensor.FromArray(Global.Device, new float[] { -1, 2, 3, -4, 5, 6, 7, -8, 9 });
            x = x.Reshape(3, -1);
            Constraints.MaxNorm maxNorm = new Constraints.MaxNorm(2, 0);
            var w = maxNorm.Call(x);
            w.Print();
        }

        [TestMethod]
        public void NonNegTest()
        {
            Tensor x = Tensor.FromArray(Global.Device, new float[] { -1, 2, 3, -4, 5, 6, 7, -8, 9 });
            x = x.Reshape(3, -1);
            Constraints.NonNeg constraint = new Constraints.NonNeg();
            var w = constraint.Call(x);
            w.Print();
        }

        [TestMethod]
        public void UnitNormTest()
        {
            Tensor x = Tensor.FromArray(Global.Device, new float[] { -1, 2, 3, -4, 5, 6, 7, -8, 9 });
            x = x.Reshape(3, -1);
            Constraints.UnitNorm constraint = new Constraints.UnitNorm(1);
            var w = constraint.Call(x);
            w.Print();
        }

        [TestMethod]
        public void MinMaxNormTest()
        {
            Tensor x = Tensor.FromArray(Global.Device, new float[] { -1, 2, 3, -4, 5, 6, 7, -8, 9 });
            x = x.Reshape(3, -1);
            Constraints.MinMaxNorm constraint = new Constraints.MinMaxNorm(0, 2, 2, null);
            var w = constraint.Call(x);
            w.Print();
        }
    }
}
