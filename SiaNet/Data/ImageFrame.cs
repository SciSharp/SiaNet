using System;
using System.Collections.Generic;
using System.Drawing;
using System.Text;
using OpenCvSharp.Extensions;
using SiaNet.Engine;

namespace SiaNet.Data
{
    public class ImageFrame : DataFrame
    {
        public ImageFrame(Tensor imageTensor)
        {
            UnderlayingTensor = imageTensor;
        }

        public ImageFrame(Bitmap img, int? width = null, int? height = null)
        {
            if (width.HasValue)
            {
                img = new Bitmap(img, new Size(width.Value, height.Value));
            }

            LoadBmp(img);
        }

        public ImageFrame(string imagePath, int? width = null, int? height = null)
        {
            Bitmap bitmap = new Bitmap(imagePath);
            if(width.HasValue)
            {
                bitmap = new Bitmap(bitmap, new Size(width.Value, height.Value));
            }

            LoadBmp(bitmap);
        }

        private void LoadBmp(Bitmap img)
        {
            var mat = img.ToMat();
            var data = Array.ConvertAll(mat.ToBytes(), x => ((float)x));
            UnderlayingTensor = K.CreateVariable(data, new long[] { 1, mat.Channels(), mat.Height, mat.Width });
        }
    }
}
