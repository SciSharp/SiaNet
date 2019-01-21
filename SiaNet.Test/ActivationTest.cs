using Microsoft.VisualStudio.TestTools.UnitTesting;
using SiaNet.Layers;
using SiaNet.Layers.Activations;
using TensorSharp;

namespace SiaNet.Test
{
    [TestClass]
    public class ActivationTest
    {
        private void RunAct(BaseLayer l, Tensor x, Tensor grad)
        {
            l.Forward(x);
            l.Output.Print();

            l.Backward(grad);
            l.Input.Grad.Print();
        }

        [TestMethod]
        public void Softmax()
        {
            Tensor x = Tensor.FromArray(Global.Device, new float[] { -1, 2, 3, -4, 5, 6, 7, -8, 9 });
            Tensor grad = Tensor.FromArray(Global.Device, new float[] { 1, -1, -2, 2, -3, -4, 5, 4, -5 });
            x = x.Reshape(3, -1);
            grad = grad.Reshape(3, -1);
            var act = new Softmax();
            RunAct(act, x, grad);
        }

        [TestMethod]
        public void Elu()
        {
            Tensor x = Tensor.FromArray(Global.Device, new float[] { -1, 2, 3, -4, 5, 6, 7, -8, 9 });
            Tensor grad = Tensor.FromArray(Global.Device, new float[] { 1, -1, -2, 2, -3, -4, 5, 4, -5 });
            x = x.Reshape(3, -1);
            grad = grad.Reshape(3, -1);
            var act = new Elu();
            RunAct(act, x, grad);
        }

        [TestMethod]
        public void Selu()
        {
            Tensor x = Tensor.FromArray(Global.Device, new float[] { -1, 2, 3, -4, 5, 6, 7, -8, 9 });
            Tensor grad = Tensor.FromArray(Global.Device, new float[] { 1, -1, -2, 2, -3, -4, 5, 4, -5 });
            x = x.Reshape(3, -1);
            grad = grad.Reshape(3, -1);
            var act = new Selu();
            RunAct(act, x, grad);
        }

        [TestMethod]
        public void Softplus()
        {
            Tensor x = Tensor.FromArray(Global.Device, new float[] { -1, 2, 3, -4, 5, 6, 7, -8, 9 });
            Tensor grad = Tensor.FromArray(Global.Device, new float[] { 1, -1, -2, 2, -3, -4, 5, 4, -5 });
            x = x.Reshape(3, -1);
            grad = grad.Reshape(3, -1);
            var act = new Softplus();
            RunAct(act, x, grad);
        }

        [TestMethod]
        public void Softsign()
        {
            Tensor x = Tensor.FromArray(Global.Device, new float[] { -1, 2, 3, -4, 5, 6, 7, -8, 9 });
            Tensor grad = Tensor.FromArray(Global.Device, new float[] { 1, -1, -2, 2, -3, -4, 5, 4, -5 });
            x = x.Reshape(3, -1);
            grad = grad.Reshape(3, -1);
            var act = new Softsign();
            RunAct(act, x, grad);
        }

        [TestMethod]
        public void Relu()
        {
            Tensor x = Tensor.FromArray(Global.Device, new float[] { -1, 2, 3, -4, 5, 6, 7, -8, 9 });
            Tensor grad = Tensor.FromArray(Global.Device, new float[] { 1, -1, -2, 2, -3, -4, 5, 4, -5 });
            x = x.Reshape(3, -1);
            grad = grad.Reshape(3, -1);
            var act = new Relu();
            RunAct(act, x, grad);
        }

        [TestMethod]
        public void Exp()
        {
            Tensor x = Tensor.FromArray(Global.Device, new float[] { -1, 2, 3, -4, 5, 6, 7, -8, 9 });
            Tensor grad = Tensor.FromArray(Global.Device, new float[] { 1, -1, -2, 2, -3, -4, 5, 4, -5 });
            x = x.Reshape(3, -1);
            grad = grad.Reshape(3, -1);
            var act = new Exp();
            RunAct(act, x, grad);
        }

        [TestMethod]
        public void Sigmoid()
        {
            Tensor x = Tensor.FromArray(Global.Device, new float[] { -1, 2, 3, -4, 5, 6, 7, -8, 9 });
            Tensor grad = Tensor.FromArray(Global.Device, new float[] { 1, -1, -2, 2, -3, -4, 5, 4, -5 });
            x = x.Reshape(3, -1);
            grad = grad.Reshape(3, -1);
            var act = new Sigmoid();
            RunAct(act, x, grad);
        }

        [TestMethod]
        public void TanH()
        {
            Tensor x = Tensor.FromArray(Global.Device, new float[] { -1, 2, 3, -4, 5, 6, 7, -8, 9 });
            Tensor grad = Tensor.FromArray(Global.Device, new float[] { 1, -1, -2, 2, -3, -4, 5, 4, -5 });
            x = x.Reshape(3, -1);
            grad = grad.Reshape(3, -1);
            var act = new Tanh();
            RunAct(act, x, grad);
        }

        [TestMethod]
        public void HardSigmoid()
        {
            Tensor x = Tensor.FromArray(Global.Device, new float[] { -1, 2, 3, -4, 5, 6, 7, -8, 9 });
            Tensor grad = Tensor.FromArray(Global.Device, new float[] { 1, -1, -2, 2, -3, -4, 5, 4, -5 });
            x = x.Reshape(3, -1);
            grad = grad.Reshape(3, -1);
            var act = new HardSigmoid();
            RunAct(act, x, grad);
        }

        [TestMethod]
        public void LeakyRelu()
        {
            Tensor x = Tensor.FromArray(Global.Device, new float[] { -1, 2, 3, -4, 5, 6, 7, -8, 9 });
            Tensor grad = Tensor.FromArray(Global.Device, new float[] { 1, -1, -2, 2, -3, -4, 5, 4, -5 });
            x = x.Reshape(3, -1);
            grad = grad.Reshape(3, -1);
            var act = new LeakyRelu();
            RunAct(act, x, grad);
        }

        [TestMethod]
        public void PRelu()
        {
            Tensor x = Tensor.FromArray(Global.Device, new float[] { -1, 2, 3, -4, 5, 6, 7, -8, 9 });
            Tensor grad = Tensor.FromArray(Global.Device, new float[] { 1, -1, -2, 2, -3, -4, 5, 4, -5 });
            x = x.Reshape(3, -1);
            grad = grad.Reshape(3, -1);
            var act = new PRelu();
            RunAct(act, x, grad);
        }
    }
}
