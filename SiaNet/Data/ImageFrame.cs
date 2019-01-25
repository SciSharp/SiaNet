using System;
using System.Collections.Generic;
using System.Drawing;
using System.Text;
using TensorSharp;
using OpenCvSharp.Extensions;

namespace SiaNet.Data
{
    public class ImageFrame : DataFrame
    {
        public ImageFrame(Tensor imageTensor)
        {
            UnderlayingTensor = imageTensor;
        }

        public ImageFrame(Bitmap img, int? height = null, int? width = null)
        {
            var mat = img.ToMat();
            UnderlayingTensor = img.ToTensor(Global.Device);
        }

        public ImageFrame(string imagePath, int? height = null, int? width = null)
        {
            Bitmap bitmap = new Bitmap(imagePath);
            UnderlayingTensor = bitmap.ToTensor(Global.Device);
        }
    }
}
