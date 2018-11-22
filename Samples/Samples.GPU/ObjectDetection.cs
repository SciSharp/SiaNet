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
    public class ObjectDetection
    {
        public static List<PredResult> PascalDetection()
        {
            string imagePath = string.Format("{0}images\\objdet.jpg", AppDomain.CurrentDomain.BaseDirectory);
            FastRCNN app = new FastRCNN(FastRCNNModel.Pascal);
            app.LoadModel();
            
            return app.Predict(imagePath);
        }

        public static List<PredResult> GroceryDetection()
        {
            string imagePath = string.Format("{0}images\\grocery.jpg", AppDomain.CurrentDomain.BaseDirectory);
            FastRCNN app = new FastRCNN(FastRCNNModel.Grocery100);
            app.LoadModel();
            return app.Predict(imagePath);
        }
    }
}
