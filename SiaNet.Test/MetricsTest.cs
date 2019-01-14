using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorSharp;

namespace SiaNet.Test
{
    [TestClass]
    public class MetricsTest
    {
        private void RunMetrics(Metrics.BaseMetric loss, Tensor preds, Tensor labels)
        {
            var w = loss.Call(preds, labels);
            var data = w.ToArray();
            w.Print();
        }

        [TestMethod]
        public void MSE()
        {
            Tensor preds = Tensor.FromArray(Global.Device, new float[] { -1, 2, 3, -4, 5, 6, 7, -8, 9 });
            preds = preds.Reshape(3, -1);

            Tensor labels = Tensor.FromArray(Global.Device, new float[] { 9, 8, 7, 6, 5, 4, 3, 2, 1 });
            labels = labels.Reshape(3, -1);
            var loss = new Metrics.MSE();
            RunMetrics(loss, preds, labels);
        }

        [TestMethod]
        public void MSLE()
        {
            Tensor preds = Tensor.FromArray(Global.Device, new float[] { -1, 2, 3, -4, 5, 6, 7, -8, 9 });
            preds = preds.Reshape(3, -1);

            Tensor labels = Tensor.FromArray(Global.Device, new float[] { 9, 8, 7, 6, 5, 4, 3, 2, 1 });
            labels = labels.Reshape(3, -1);

            var loss = new Metrics.MSLE();
            RunMetrics(loss, preds, labels);
        }

        [TestMethod]
        public void MAE()
        {
            Tensor preds = Tensor.FromArray(Global.Device, new float[] { -1, 2, 3, -4, 5, 6, 7, -8, 9 });
            preds = preds.Reshape(3, -1);

            Tensor labels = Tensor.FromArray(Global.Device, new float[] { 9, 8, 7, 6, 5, 4, 3, 2, 1 });
            labels = labels.Reshape(3, -1);

            var loss = new Metrics.MAE();
            RunMetrics(loss, preds, labels);

        }

        [TestMethod]
        public void MAPE()
        {
            Tensor preds = Tensor.FromArray(Global.Device, new float[] { -1, 2, 3, -4, 5, 6, 7, -8, 9 });
            preds = preds.Reshape(3, -1);

            Tensor labels = Tensor.FromArray(Global.Device, new float[] { 9, 8, 7, 6, 5, 4, 3, 2, 1 });
            labels = labels.Reshape(3, -1);

            var loss = new Metrics.MAPE();
            RunMetrics(loss, preds, labels);

        }

        [TestMethod]
        public void Accuracy()
        {
            Tensor preds = Tensor.FromArray(Global.Device, new float[] { 0.7f, 0.1f, 0.6f, 0.9f, 0.55f, 0.05f, 0.9f, 0.01f, 0.09f });
            preds = preds.Reshape(-1, 3);

            Tensor labels = Tensor.FromArray(Global.Device, new float[] { 1, 0, 0, 0, 0, 1, 0, 1, 0 });
            labels = labels.Reshape(-1, 3);

            var loss = new Metrics.Accuracy();
            RunMetrics(loss, preds, labels);
        }

    }
}
