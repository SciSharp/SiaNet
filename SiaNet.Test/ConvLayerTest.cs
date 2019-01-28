using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using TensorSharp;

namespace SiaNet.Test
{
    [TestClass]
    public class ConvLayerTest
    {
        [TestMethod]
        public void Conv2D()
        {
            Global.UseGpu();
            Tensor x = Tensor.FromArray(Global.Device, new float[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 8, 7, 6, 5, 4, 3, 2, 1 });
            x = x.Reshape(2, 1, 3, 3);
         
            Tensor grad = Tensor.FromArray(Global.Device, new float[] { 1, -1, -2, 2, -3, -4, 5, 4, -5, 1, -1, -2, 2, -3, -4, 5, 4, -5 });
            grad = grad.Reshape(3, -1);
            var l = new Layers.Conv2D(3, kernalSize: Tuple.Create<uint, uint>(3, 3), kernalInitializer: new Initializers.Ones(), padding: PaddingType.Same);
            l.Forward(x);
            l.Output.Print();
            //l.Backward(grad);
            //l.Input.Grad.Print();
        }
       
    }
}
