using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorSharp;

namespace SiaNet.Test
{
    [TestClass]
    public class LossFnTest
    {
        private void RunLossFn(Losses.BaseLoss loss, Tensor preds, Tensor labels)
        {
            var w = loss.Call(preds, labels);
            var data = w.ToArray();
            w.Print();

            var grad = loss.CalcGrad(preds, labels);
            grad.Print();
        }

        [TestMethod]
        public void MSE()
        {
            Tensor preds = Tensor.FromArray(Global.Device, new float[] { -1, 2, 3, -4, 5, 6, 7, -8, 9 });
            preds = preds.Reshape(3, -1);

            Tensor labels = Tensor.FromArray(Global.Device, new float[] { 9, 8, 7, 6, 5, 4, 3, 2, 1 });
            labels = labels.Reshape(3, -1);

            var loss = new Losses.MeanSquaredError();
            RunLossFn(loss, preds, labels);
        }

        [TestMethod]
        public void MSLE()
        {
            Tensor preds = Tensor.FromArray(Global.Device, new float[] { -1, 2, 3, -4, 5, 6, 7, -8, 9 });
            preds = preds.Reshape(3, -1);

            Tensor labels = Tensor.FromArray(Global.Device, new float[] { 9, 8, 7, 6, 5, 4, 3, 2, 1 });
            labels = labels.Reshape(3, -1);

            var loss = new Losses.MeanSquaredLogError();
            RunLossFn(loss, preds, labels);
        }

        [TestMethod]
        public void MAE()
        {
            Tensor preds = Tensor.FromArray(Global.Device, new float[] { -1, 2, 3, -4, 5, 6, 7, -8, 9 });
            preds = preds.Reshape(3, -1);

            Tensor labels = Tensor.FromArray(Global.Device, new float[] { 9, 8, 7, 6, 5, 4, 3, 2, 1 });
            labels = labels.Reshape(3, -1);

            var loss = new Losses.MeanAbsoluteError();
            RunLossFn(loss, preds, labels);

        }

        [TestMethod]
        public void MAPE()
        {
            Tensor preds = Tensor.FromArray(Global.Device, new float[] { -1, 2, 3, -4, 5, 6, 7, -8, 9 });
            preds = preds.Reshape(3, -1);

            Tensor labels = Tensor.FromArray(Global.Device, new float[] { 9, 8, 7, 6, 5, 4, 3, 2, 1 });
            labels = labels.Reshape(3, -1);

            var loss = new Losses.MeanAbsolutePercentageError();
            RunLossFn(loss, preds, labels);

        }

        [TestMethod]
        public void SquareHinge()
        {
            Tensor preds = Tensor.FromArray(Global.Device, new float[] { -1, 2, 3, -4, 5, 6, 7, -8, 9 });
            preds = preds.Reshape(3, -1);

            Tensor labels = Tensor.FromArray(Global.Device, new float[] { 9, 8, 7, 6, 5, 4, 3, 2, 1 });
            labels = labels.Reshape(3, -1);

            var loss = new Losses.SquaredHinge();
            RunLossFn(loss, preds, labels);

        }

        [TestMethod]
        public void Hinge()
        {
            Tensor preds = Tensor.FromArray(Global.Device, new float[] { -1, 2, 3, -4, 5, 6, 7, -8, 9 });
            preds = preds.Reshape(3, -1);

            Tensor labels = Tensor.FromArray(Global.Device, new float[] { 9, 8, 7, 6, 5, 4, 3, 2, 1 });
            labels = labels.Reshape(3, -1);

            var loss = new Losses.Hinge();
            RunLossFn(loss, preds, labels);

        }

        [TestMethod]
        public void CategorialHinge()
        {
            Tensor preds = Tensor.FromArray(Global.Device, new float[] { -1, 2, 3, -4, 5, 6, 7, -8, 9 });
            preds = preds.Reshape(3, -1);

            Tensor labels = Tensor.FromArray(Global.Device, new float[] { 9, 8, 7, 6, 5, 4, 3, 2, 1 });
            labels = labels.Reshape(3, -1);

            var loss = new Losses.CategorialHinge();
            RunLossFn(loss, preds, labels);
        }

        [TestMethod]
        public void LogCosh()
        {
            Tensor preds = Tensor.FromArray(Global.Device, new float[] { -1, 2, 3, -4, 5, 6, 7, -8, 9 });
            preds = preds.Reshape(3, -1);

            Tensor labels = Tensor.FromArray(Global.Device, new float[] { 9, 8, 7, 6, 5, 4, 3, 2, 1 });
            labels = labels.Reshape(3, -1);

            var loss = new Losses.LogCosh();
            RunLossFn(loss, preds, labels);
        }

        [TestMethod]
        public void KullbackLeiblerDivergence()
        {
            Tensor preds = Tensor.FromArray(Global.Device, new float[] { -1, 2, 3, -4, 5, 6, 7, -8, 9 });
            preds = preds.Reshape(3, -1);

            Tensor labels = Tensor.FromArray(Global.Device, new float[] { 9, 8, 7, 6, 5, 4, 3, 2, 1 });
            labels = labels.Reshape(3, -1);

            var loss = new Losses.KullbackLeiblerDivergence();
            RunLossFn(loss, preds, labels);
        }

        [TestMethod]
        public void Poisson()
        {
            Tensor preds = Tensor.FromArray(Global.Device, new float[] { 9, 8, 7, 6, 5, 4, 3, 2, 1 });
            preds = preds.Reshape(3, -1);

            Tensor labels = Tensor.FromArray(Global.Device, new float[] { -1, 2, 3, -4, 5, 6, 7, -8, 9 });
            labels = labels.Reshape(3, -1);

            var loss = new Losses.Poisson();
            RunLossFn(loss, preds, labels);
        }

        [TestMethod]
        public void CosineProximity()
        {
            Tensor preds = Tensor.FromArray(Global.Device, new float[] { 9, 8, 7, 6, 5, 4, 3, 2, 1 });
            preds = preds.Reshape(3, -1);

            Tensor labels = Tensor.FromArray(Global.Device, new float[] { -1, 2, 3, -4, 5, 6, 7, -8, 9 });
            labels = labels.Reshape(3, -1);

            var loss = new Losses.CosineProximity();
            RunLossFn(loss, preds, labels);
        }
    }
}
