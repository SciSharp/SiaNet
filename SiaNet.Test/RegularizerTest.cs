using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorSharp;

namespace SiaNet.Test
{
    [TestClass]
    public class RegularizerTest
    {
        [TestMethod]
        public void L1Test()
        {
            Tensor x = Tensor.FromArray(Global.Device, new float[] { -1, 2, 3, -4, 5, 6, 7, -8, 9 });
            x = x.Reshape(3, -1);
            Regularizers.L1 reg = new Regularizers.L1();
            var w = reg.Call(x);

            var grad = reg.CalcGrad(x);
            grad.Print();
        }


        [TestMethod]
        public void L2Test()
        {
            Tensor x = Tensor.FromArray(Global.Device, new float[] { -1, 2, 3, -4, 5, 6, 7, -8, 9 });
            x = x.Reshape(3, -1);
            Regularizers.L2 reg = new Regularizers.L2();
            var w = reg.Call(x);

            var grad = reg.CalcGrad(x);
            grad.Print();
        }

        [TestMethod]
        public void L1L2Test()
        {
            Tensor x = Tensor.FromArray(Global.Device, new float[] { -1, 2, 3, -4, 5, 6, 7, -8, 9 });
            x = x.Reshape(3, -1);
            Regularizers.L1L2 reg = new Regularizers.L1L2();
            var w = reg.Call(x);

            var grad = reg.CalcGrad(x);
            grad.Print();
        }
    }
}
