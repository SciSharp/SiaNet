using CNTK;
using Emgu.CV.Structure;
using SiaNet.Processing;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SiaNet.Model
{
    internal class ImageMapInfo
    {
        internal string Filepath;
        internal int Label;
        internal int RotationAngle;
        internal Emgu.CV.CvEnum.FlipType Flip = Emgu.CV.CvEnum.FlipType.None;
        internal int Resize = 0;
    }

    public class ImageDataFrame : XYFrame
    {
        private int[] features;
        private int labels;
        private string folder;
        private bool fromFolder;
        List<ImageMapInfo> folderMapData;

        public ImageDataFrame(Variable feature, Variable label)
        {
            features = feature.Shape.Dimensions.ToArray();
            labels = label.Shape.Dimensions[0];
            
            counter = 0;
        }

        public ImageDataFrame(Variable feature, Variable label, string folder, int resize = 0, int numberOfRandomRotation = 0, bool horizontalFlip = false, bool verticalFlip = false)
            : this(feature, label)
        {
            this.folder = folder;
            fromFolder = true;
            folderMapData = new List<ImageMapInfo>();
            DirectoryInfo dir = new DirectoryInfo(folder);
            var subfolders = dir.GetDirectories();
            int counter = 1;
            foreach (var item in subfolders)
            {
                var files = item.GetFiles().Select(x => (x.FullName)).ToList();
                foreach (var file in files)
                {
                    folderMapData.Add(new ImageMapInfo() { Filepath = file, Label = counter, RotationAngle = 0, Resize = resize });
                    if (numberOfRandomRotation > 0)
                    {
                        for (int i = 0; i < numberOfRandomRotation; i++)
                        {
                            folderMapData.Add(new ImageMapInfo() { Filepath = file, Label = counter, RotationAngle = new Random(30).Next(10, 360), Resize = resize });
                        }
                    }

                    if (horizontalFlip)
                    {
                        folderMapData.Add(new ImageMapInfo() { Filepath = file, Label = counter, RotationAngle = 0, Flip = Emgu.CV.CvEnum.FlipType.Horizontal, Resize = resize });
                    }

                    if (verticalFlip)
                    {
                        folderMapData.Add(new ImageMapInfo() { Filepath = file, Label = counter, RotationAngle = 0, Flip = Emgu.CV.CvEnum.FlipType.Vertical, Resize = resize });
                    }
                }

                counter++;
            }

            Shuffle();
        }

        private int counter;

        internal Value CurrentX { get; set; }

        internal Value CurrentY { get; set; }

        internal bool Next(int batchSize)
        {
            bool result = true;
            if(fromFolder)
            {
                result = GetNextFromFolder(batchSize);
            }
            else
            {
                result = GetNextFromFrame(batchSize);
            }

            return true;
        }

        private bool GetNextFromFrame(int batchSize)
        {
            var batchData = XFrame.Data.Skip(counter * batchSize).Take(batchSize).ToList();
            
            if (batchData.Count == 0)
                return false;

            List<byte> byteData = new List<byte>();
            List<byte> labelData = new List<byte>();

            //foreach (var cur in batchData)
            //{
            //    foreach (var item in cur.Pixels)
            //    {
            //        foreach (var i in item)
            //        {
            //            byteData.AddRange(i);
            //        }
            //    }

            //    for (int i = 1; i <= labels; i++)
            //    {
            //        if (cur.label == i)
            //        {
            //            labelData.Add(1);
            //        }
            //        else
            //        {
            //            labelData.Add(0);
            //        }
            //    }
            //}
             
            CurrentX = Value.CreateBatch(features, byteData.Select(b => (float)b).ToList(), GlobalParameters.Device);
            CurrentY = Value.CreateBatch(features, labelData.Select(b => (float)b).ToList(), GlobalParameters.Device);

            return true;
        }

        public bool GetNextFromFolder(int batchSize)
        {
            var batchData = folderMapData.Skip(counter * batchSize).Take(batchSize).ToList();
            if (batchData.Count == 0)
                return false;

            List<float> byteData = new List<float>();
            List<byte> labelData = new List<byte>();

            foreach (var item in batchData)
            {
                byteData.AddRange(processImageFile(item));

                for (int i = 1; i <= labels; i++)
                {
                    if (item.Label == i)
                    {
                        labelData.Add(1);
                    }
                    else
                    {
                        labelData.Add(0);
                    }
                }
            }

            CurrentX = Value.CreateBatch(features, byteData, GlobalParameters.Device);
            CurrentY = Value.CreateBatch(features, labelData.Select(b => (float)b).ToList(), GlobalParameters.Device);

            return true;
        }

        private List<float> processImageFile(ImageMapInfo mapInfo)
        {
            Bitmap bmp = new Bitmap(mapInfo.Filepath);
            Emgu.CV.Image<Bgr, byte> img = new Emgu.CV.Image<Bgr, byte>(bmp);
            if (mapInfo.Resize > 0)
            {
                img = img.Resize(mapInfo.Resize, mapInfo.Resize, Emgu.CV.CvEnum.Inter.Nearest);
            }

            if (mapInfo.Flip != Emgu.CV.CvEnum.FlipType.None)
            {
                img = img.Flip(mapInfo.Flip);
            }

            if (mapInfo.RotationAngle > 0)
            {
                img.Rotate(mapInfo.RotationAngle, new Bgr(Color.White));
            }

            return img.Bitmap.ParallelExtractCHW();
        }

        internal void Reset()
        {
            counter = 1;
            CurrentX = null;
            CurrentY = null;
        }

        private void Shuffle()
        {
            List<ImageMapInfo> clone = folderMapData;
            if (folderMapData.Count > 0)
            {
                clone.Clear();
                Random random = new Random();

                while (folderMapData.Count > 0)
                {
                    int row = random.Next(0, folderMapData.Count);
                    var element = folderMapData.ElementAt(row);
                    clone.Add(element);
                    folderMapData.Remove(element);
                }
            }

            folderMapData = clone;
        }

        public void ExtractCifar10()
        {
            string filepath = @"C:\Users\batt0153\AppData\Local\Downloads\cifar-10-binary\cifar-10-batches-bin\data_batch_1.bin";
            FileStream imageStream = new FileStream(filepath, FileMode.Open);
            BinaryReader br = new BinaryReader(imageStream);
            int pixelSize = 32 * 32 * 3;
            
            List<byte> imageRec = null;
            
            for (int di = 0; di < 10000; ++di)
            {
                imageRec = new List<byte>();
                float lbl = br.ReadByte();

                imageRec.AddRange(br.ReadBytes(pixelSize));
                XFrame.Data.Add(imageRec.Select(x => ((float)x)).ToList());
                YFrame.Data.Add(new List<float>() { lbl });
            }

            imageStream.Close();
        }
    }
}
