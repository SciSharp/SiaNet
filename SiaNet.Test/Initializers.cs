using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorSharp;

namespace SiaNet.Test
{
    [TestClass]
    public class InitializersTest
    {
        [TestMethod]
        public void ConstantTest()
        {
            RunInit(new Initializers.Constant(2));
        }

        [TestMethod]
        public void OnesTest()
        {
            RunInit(new Initializers.Ones());
        }

        [TestMethod]
        public void ZerosTest()
        {
            RunInit(new Initializers.Zeros());
        }

        [TestMethod]
        public void GlorotUniformTest()
        {
            RunInit(new Initializers.GlorotUniform(7));
        }

        [TestMethod]
        public void GlorotNormalTest()
        {
            RunInit(new Initializers.GlorotNormal(7));
        }

        [TestMethod]
        public void HeNormalTest()
        {
            RunInit(new Initializers.HeNormal(7));
        }

        [TestMethod]
        public void HeUniformTest()
        {
            RunInit(new Initializers.HeUniform(7));
        }

        [TestMethod]
        public void LecunnNormalTest()
        {
            RunInit(new Initializers.LecunNormal(7));
        }

        [TestMethod]
        public void LecunnUniformTest()
        {
            RunInit(new Initializers.LecunUniform(7));
        }

        [TestMethod]
        public void RandomNormalTest()
        {
            RunInit(new Initializers.RandomNormal(7));
        }

        [TestMethod]
        public void RandomUniformTest()
        {
            RunInit(new Initializers.RandomUniform(7));
        }

        private void RunInit(Initializers.BaseInitializer initializer)
        {
            Tensor x = new Tensor(Global.Device, DType.Float32, 3, 3);
            x = initializer.Operator(x.Shape);
            x.Print();
        }
    }
}
