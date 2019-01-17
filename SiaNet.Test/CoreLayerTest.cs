using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorSharp;

namespace SiaNet.Test
{
    [TestClass]
    public class CoreLayerTest
    {
        [TestMethod]
        public void Dense()
        {
            Tensor x = Tensor.FromArray(Global.Device, new float[] { -1, 2, 3, -4, 5, 6, 7, -8, 9 });
            x = x.Reshape(3, -1);
            Tensor grad = Tensor.FromArray(Global.Device, new float[] { 1, -1, -2, 2, -3, -4, 5, 4, -5 });
            grad = grad.Reshape(3, -1);
            Layers.Dense l = new Layers.Dense(3, ActivationType.Linear, new Initializers.Ones(), useBias: true);
            l.Forward(Parameter.Create(x));
            l.Output.Print();
            l.Backward(grad);
            l.Input.Grad.Print();
        }
       
    }
}
