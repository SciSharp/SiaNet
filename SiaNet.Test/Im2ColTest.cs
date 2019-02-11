using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorSharp;
using SiaNet;
using System;

namespace SiaNet.Test
{
    [TestClass]
    public class Im2ColTest
    {
        [TestMethod]
        public void Im2Col_2d()
        {
            Global.UseGpu();
            Tensor x = Tensor.FromArray(Global.Device, new float[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 8, 7, 6, 5, 4, 3, 2, 1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 8, 7, 6, 5, 4, 3, 2, 1 });
            x = x.Reshape(2, 2, 3, 3);
            var cols = ImgUtil.Im2Col(x, Tuple.Create<uint, uint>(3, 3), 1, 1);
            cols.Print();

            var im = ImgUtil.Col2Im(cols, x.Shape, Tuple.Create<uint, uint>(3, 3), 1, 1);
            im.Print();
        }

        [TestMethod]
        public void DiagTest()
        {
            Global.UseGpu();
            Tensor x = Tensor.FromArray(Global.Device, new float[] { 1, 2, 3, 4, 5, 6, 7, 8, 9 });
            x = x.Reshape(3, 3);

            var result = TOps.Diag(x);
            result.Print();
        }
    }
}
