using SiaNet.Application;
using SiaNet.Common;
using SiaNet.Model;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Samples.GPU
{
    public class ImageClassification
    {
        public static List<PredResult> ImagenetTest(ImageNetModel model)
        {
            string imagePath = string.Format("{0}images\\dog_cls.jpg", AppDomain.CurrentDomain.BaseDirectory);
            ImageNet app = new ImageNet(model);
            app.LoadModel();
            return app.Predict(imagePath);
        }

        public static List<PredResult> Cifar10Test(Cifar10Model model)
        {
            string imagePath = string.Format("{0}images\\dog_cls.jpg", AppDomain.CurrentDomain.BaseDirectory);
            Cifar10 app = new Cifar10(model);
            app.LoadModel();
            return app.Predict(imagePath);
        }
    }
}
