using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using SiaNet.Model;
using SiaNet.Common;
using System.IO;
using System.IO.Compression;

namespace DatasetExtractor
{
    internal class MNISTExtractor : BaseExtractor, IExtractor
    {
        string folder = string.Format("{0}\\MNIST", DefaultPath.Datasets);

        string[] urls = new string[] 
        {
            "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz",
            "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz",
            "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz",
            "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz"
        };

        public XYFrame TrainFrame { get; set; }

        public XYFrame TestFrame { get; set; }

        public void Download()
        {
            string tempFolder = folder + "\\tmp";
            if (!Directory.Exists(tempFolder))
            {
                Directory.CreateDirectory(tempFolder);
            }

            foreach (var item in urls)
            {
                string filename = Path.GetFileName(item);
                string fullpath = tempFolder + "\\" + filename;

                if (File.Exists(fullpath))
                {
                    continue;
                }

                DownloadFile(item, fullpath);
            }
        }

        public void Extract()
        {
            string tempFolder = folder + "\\tmp";
            TrainFrame = new XYFrame();
            TestFrame = new XYFrame();
            ExtractTrain(tempFolder);
            ExtractTest(tempFolder);

            TrainFrame.Save(folder + "\\train");
            TestFrame.Save(folder + "\\test");

            Directory.Delete(tempFolder, true);
        }

        private void ExtractTrain(string tmpfolder)
        {
            string trainImages = tmpfolder + "\\train-images-idx3-ubyte.gz";
            string trainLabels = tmpfolder + "\\train-labels-idx1-ubyte.gz";

            GZipStream imageStream = new GZipStream(new FileStream(trainImages, FileMode.Open), CompressionMode.Decompress);
            GZipStream labelStream = new GZipStream(new FileStream(trainLabels, FileMode.Open), CompressionMode.Decompress);
            BinaryReader brimg = new BinaryReader(imageStream);
            BinaryReader brlbl = new BinaryReader(labelStream);
            int magic1 = brimg.ReadInt32(); // discard
            int numImages = brimg.ReadInt32();
            int numRows = brimg.ReadInt32();
            int numCols = brimg.ReadInt32();

            int magic2 = brlbl.ReadInt32();
            int numLabels = brlbl.ReadInt32();
            int pixelSize = 28 * 28 * 1;
            List<byte> imageData = null;

            for (int di = 0; di < 60000; ++di)
            {
                imageData = new List<byte>();
                imageData.AddRange(brimg.ReadBytes(pixelSize));

                float lbl = brlbl.ReadByte();

                TrainFrame.XFrame.Data.Add(imageData.Select(x => ((float)x)).ToList());
                TrainFrame.YFrame.Data.Add(new List<float>() { lbl });
            }

            imageStream.Close();
            labelStream.Close();
        }

        private void ExtractTest(string tmpfolder)
        {
            string trainImages = tmpfolder + "\\t10k-images-idx3-ubyte.gz";
            string trainLabels = tmpfolder + "\\t10k-labels-idx1-ubyte.gz";

            GZipStream imageStream = new GZipStream(new FileStream(trainImages, FileMode.Open), CompressionMode.Decompress);
            GZipStream labelStream = new GZipStream(new FileStream(trainLabels, FileMode.Open), CompressionMode.Decompress);
            BinaryReader brimg = new BinaryReader(imageStream);
            BinaryReader brlbl = new BinaryReader(labelStream);
            int magic1 = brimg.ReadInt32(); // discard
            int numImages = brimg.ReadInt32();
            int numRows = brimg.ReadInt32();
            int numCols = brimg.ReadInt32();

            int magic2 = brlbl.ReadInt32();
            int numLabels = brlbl.ReadInt32();
            int pixelSize = 28 * 28 * 1;
            List<byte> imageData = null;

            for (int di = 0; di < 10000; ++di)
            {
                imageData = new List<byte>();
                imageData.AddRange(brimg.ReadBytes(pixelSize));

                float lbl = brlbl.ReadByte();

                TestFrame.XFrame.Data.Add(imageData.Select(x => ((float)x)).ToList());
                TestFrame.YFrame.Data.Add(new List<float>() { lbl });
            }

            imageStream.Close();
            labelStream.Close();
        }
    }
}
