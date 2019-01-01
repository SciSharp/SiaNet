using System;
using System.Collections.Generic;
using System.Text;
using TensorSharp;

namespace SiaNet.Data
{
    public class ImageFrame : DataFrame
    {
        public ImageFrame(params long[] shape)
            : base(shape)
        {

        }

        public ImageFrame(Tensor imageTensor)
        {
            Shape = imageTensor.Shape;
            underlayingVariable = imageTensor;
        }
    }
}
