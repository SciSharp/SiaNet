namespace SiaNet.Data
{
    using System;
    using System.Drawing;
    using OpenCvSharp.Extensions;
    using SiaNet.Engine;

    /// <summary>
    /// Image dataframe to hold images
    /// </summary>
    /// <seealso cref="SiaNet.Data.DataFrame" />
    public class ImageFrame : DataFrame
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="ImageFrame"/> class.
        /// </summary>
        /// <param name="imageTensor">The image tensor.</param>
        public ImageFrame(Tensor imageTensor)
        {
            UnderlayingTensor = imageTensor;
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="ImageFrame"/> class.
        /// </summary>
        /// <param name="img">The img in bitmap format.</param>
        /// <param name="width">The width to resize the image.</param>
        /// <param name="height">The height to resize the image.</param>
        public ImageFrame(Bitmap img, int? width = null, int? height = null)
        {
            if (width.HasValue)
            {
                img = new Bitmap(img, new Size(width.Value, height.Value));
            }

            LoadBmp(img);
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="ImageFrame"/> class.
        /// </summary>
        /// <param name="imagePath">The image path.</param>
        /// <param name="width">The width to resize the image.</param>
        /// <param name="height">The height to resize the image.</param>
        public ImageFrame(string imagePath, int? width = null, int? height = null)
        {
            Bitmap bitmap = new Bitmap(imagePath);
            if(width.HasValue)
            {
                bitmap = new Bitmap(bitmap, new Size(width.Value, height.Value));
            }

            LoadBmp(bitmap);
        }

        /// <summary>
        /// Loads the BMP.
        /// </summary>
        /// <param name="img">The img.</param>
        private void LoadBmp(Bitmap img)
        {
            var mat = img.ToMat();
            var data = Array.ConvertAll(mat.ToBytes(), x => ((float)x));
            UnderlayingTensor = K.CreateVariable(data, new long[] { 1, mat.Channels(), mat.Height, mat.Width });
        }
    }
}
