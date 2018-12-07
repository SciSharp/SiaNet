using SiaNet.Backend;
using System;
using System.Collections.Generic;
using System.Text;

namespace SiaNet.Data
{
    public class DataSetParser
    {
        public static ValueTuple<DataIter, DataIter> MNIST(string trainImagesPath, string trainLabelPath, string valImagesPath, string valLabelPath, int batch_size=32)
        {
            var trainIter = new MXDataIter("MNISTIter")
                .SetParam("image", trainImagesPath)
                .SetParam("label", trainLabelPath)
                .SetParam("flat", 1)
                .SetParam("batch_size", batch_size)
                .CreateDataIter();

            var valIter = new MXDataIter("MNISTIter")
                .SetParam("image", valImagesPath)
                .SetParam("label", valLabelPath)
                .SetParam("flat", 1)
                .SetParam("batch_size", batch_size)
                .CreateDataIter();

            return ValueTuple.Create(trainIter, valIter);
        }
    }
}
