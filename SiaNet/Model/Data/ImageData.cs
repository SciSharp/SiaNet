using System;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Drawing.Imaging;

namespace SiaNet.Model.Data
{
    public class ImageData
    {
        public ImageData(
            Bitmap bitmap,
            Size canvasSize = default(Size),
            bool forceGrayScale = false,
            bool forceIgnoreAlpha = false)
        {
            Bitmap = bitmap;
            ColorPalette = bitmap.Palette.Entries;
            CanvasSize = canvasSize.IsEmpty ? Bitmap.Size : canvasSize;
            GrayScale = forceGrayScale;
            IgnoreAlpha = forceIgnoreAlpha;
        }

        public Bitmap Bitmap { get; }

        public Size CanvasSize { get; set; }

        protected Color[] ColorPalette { get; }

        public virtual Shape DataShape
        {
            get
            {
                var outputSize = CanvasSize;
                int arrayFpp;

                switch (Bitmap.PixelFormat)
                {
                    case PixelFormat.Format24bppRgb:
                    case PixelFormat.Format32bppRgb:
                        arrayFpp = GrayScale ? 1 : 3;

                        break;
                    case PixelFormat.Format32bppArgb:
                    case PixelFormat.Format8bppIndexed:
                        arrayFpp = GrayScale ? (IgnoreAlpha ? 1 : 2) : (IgnoreAlpha ? 3 : 4);

                        break;
                    default:

                        throw new NotSupportedException(string.Format(
                            "{0} is not supported as the bitmap pixel format.",
                            Bitmap.PixelFormat));
                }

                return new Shape(outputSize.Height, outputSize.Width, arrayFpp);
            }
        }

        public bool GrayScale { get; set; }

        public bool IgnoreAlpha { get; set; }

        public Matrix TransformationMatrix { get; set; } = new Matrix();

        public virtual Bitmap AsBitmap()
        {
            var bitmapFloats = AsData();
            var bitmapSize = CanvasSize;

            var bitmap = new Bitmap(bitmapSize.Width, bitmapSize.Height, PixelFormat.Format32bppArgb);

            for (var y = 0; y < bitmapSize.Height; y++)
            {
                for (var x = 0; x < bitmapSize.Width; x++)
                {
                    var colorData = bitmapFloats[y][x];
                    Color color;

                    switch (colorData.Length)
                    {
                        case 1:
                            color = Color.FromArgb((int) colorData[0], (int) colorData[0], (int) colorData[0]);

                            break;
                        case 2:
                            color = Color.FromArgb((int) colorData[1], (int) colorData[0], (int) colorData[0],
                                (int) colorData[0]);

                            break;
                        case 3:
                            color = Color.FromArgb((int) colorData[0], (int) colorData[1], (int) colorData[2]);

                            break;
                        case 4:
                            color = Color.FromArgb((int) colorData[3], (int) colorData[0], (int) colorData[1],
                                (int) colorData[2]);

                            break;
                        default:
                            color = Color.Black;

                            break;
                    }

                    bitmap.SetPixel(x, y, color);
                }
            }

            return bitmap;
        }

        public virtual unsafe float[][][] AsData()
        {
            var pixelReader = GetPixelReader();

            var bitmapData =
                Bitmap.LockBits(new Rectangle(0, 0, Bitmap.Width, Bitmap.Height),
                    ImageLockMode.ReadOnly, Bitmap.PixelFormat);

            var invertMatrix = TransformationMatrix.Clone();
            invertMatrix.Invert();
            var matrix = invertMatrix.Elements;

            var outputShape = DataShape;
            var resultArray = new float[outputShape[0]][][];

            for (var y = 0; y < outputShape[0]; y++)
            {
                resultArray[y] = new float[outputShape[1]][];

                for (var x = 0; x < outputShape[1]; x++)
                {
                    //var point = new[] {new PointF(x, y)};
                    //invertMatrix.TransformPoints(point);
                    //var projectedX = (int)point[0].X;
                    //var projectedY = (int)point[0].Y;

                    // Faster transform
                    var projectedX = (int) (x * matrix[0] + y * matrix[1] + matrix[4]);
                    var projectedY = (int) (x * matrix[2] + y * matrix[3] + matrix[5]);

                    if (projectedX < 0 ||
                        projectedY < 0 ||
                        projectedX >= bitmapData.Width ||
                        projectedY >= bitmapData.Height)
                    {
                        resultArray[y][x] = new float[outputShape[2]];

                        continue;
                    }

                    var pixelOffset = projectedY * bitmapData.Stride + projectedX * pixelReader.Item1;
                    resultArray[y][x] = pixelReader.Item2((byte*) bitmapData.Scan0 + pixelOffset);
                }
            }

            Bitmap.UnlockBits(bitmapData);

            return resultArray;
        }

        public void FlipHorizontally(float deg, PointF center)
        {
            var e = TransformationMatrix.Elements;
            TransformationMatrix = new Matrix(-e[0], e[1], e[2], e[3], e[4], e[5]);
        }

        public void FlipVertically(float deg, PointF center)
        {
            var e = TransformationMatrix.Elements;
            TransformationMatrix = new Matrix(e[0], e[1], e[2], -e[3], e[4], e[5]);
        }

        public void Rotate(float deg, PointF point)
        {
            TransformationMatrix.RotateAt(deg, point);
        }

        public void Scale(float sx, float sy)
        {
            TransformationMatrix.Scale(sx, sy);
        }

        public void Translate(float x, float y)
        {
            TransformationMatrix.Translate(x, y);
        }

        protected virtual unsafe Tuple<int, PixelReaderFunction> GetPixelReader()
        {
            int bitmapBpp;
            PixelReaderFunction pixelReaderFunction;

            switch (Bitmap.PixelFormat)
            {
                case PixelFormat.Format24bppRgb:
                    bitmapBpp = 3;
                    pixelReaderFunction = GrayScale ? (PixelReaderFunction) ReadPixelAsGrayScale : ReadPixel;

                    break;
                case PixelFormat.Format32bppRgb:
                    bitmapBpp = 4;
                    pixelReaderFunction = GrayScale ? (PixelReaderFunction) ReadPixelAsGrayScale : ReadPixel;

                    break;
                case PixelFormat.Format32bppArgb:
                    bitmapBpp = 4;

                    if (GrayScale)
                    {
                        pixelReaderFunction = IgnoreAlpha
                            ? (PixelReaderFunction) ReadPixelAsGrayScale
                            : ReadPixelAsGrayScaleWithAlpha;
                    }
                    else
                    {
                        pixelReaderFunction = IgnoreAlpha ? (PixelReaderFunction) ReadPixel : ReadPixelWithAlpha;
                    }

                    break;
                case PixelFormat.Format8bppIndexed:
                    bitmapBpp = 1;

                    if (GrayScale)
                    {
                        pixelReaderFunction = IgnoreAlpha
                            ? (PixelReaderFunction) ReadIndexedPixelAsGrayScale
                            : ReadIndexedPixelAsGrayScaleWithAlpha;
                    }
                    else
                    {
                        pixelReaderFunction =
                            IgnoreAlpha ? (PixelReaderFunction) ReadIndexedPixel : ReadIndexedPixelWithAlpha;
                    }

                    break;
                default:

                    throw new NotSupportedException(string.Format(
                        "{0} is not supported as the bitmap pixel format.",
                        Bitmap.PixelFormat));
            }

            return new Tuple<int, PixelReaderFunction>(bitmapBpp, pixelReaderFunction);
        }

        protected virtual unsafe float[] ReadIndexedPixel(byte* pointer)
        {
            var color = ColorPalette[pointer[0]];

            return new[] {color.R, color.G, (float) color.B};
        }

        protected virtual unsafe float[] ReadIndexedPixelAsGrayScale(byte* pointer)
        {
            var color = ColorPalette[pointer[0]];

            return new[] {(color.R + color.G + color.B) / 3f};
        }

        protected virtual unsafe float[] ReadIndexedPixelAsGrayScaleWithAlpha(byte* pointer)
        {
            var color = ColorPalette[pointer[0]];

            return new[] {(color.R + color.G + color.B) / 3f, color.A};
        }

        protected virtual unsafe float[] ReadIndexedPixelWithAlpha(byte* pointer)
        {
            var color = ColorPalette[pointer[0]];

            return new[] {color.R, color.G, color.B, (float) color.A};
        }

        protected virtual unsafe float[] ReadPixel(byte* pointer)
        {
            return new[] {pointer[2], pointer[1], (float) pointer[0]};
        }

        protected virtual unsafe float[] ReadPixelAsGrayScale(byte* pointer)
        {
            return new[] {(pointer[2] + pointer[1] + pointer[0]) / 3f};
        }

        protected virtual unsafe float[] ReadPixelAsGrayScaleWithAlpha(byte* pointer)
        {
            return new[] {(pointer[2] + pointer[1] + pointer[0]) / 3f, pointer[3]};
        }


        protected virtual unsafe float[] ReadPixelWithAlpha(byte* pointer)
        {
            return new[] {pointer[2], pointer[1], pointer[0], (float) pointer[3]};
        }

        protected unsafe delegate float[] PixelReaderFunction(byte* pointer);
    }
}