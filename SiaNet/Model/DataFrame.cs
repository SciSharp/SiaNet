namespace SiaNet.Model
{
    using CNTK;
    using System;
    using System.Collections.Generic;
    using System.Data;
    using System.IO;
    using System.Linq;
    using System.Text;
    using SiaNet.Processing;
    using System.Threading.Tasks;
    using Microsoft.VisualBasic.FileIO;
    using System.Runtime.Serialization;
    using System.Runtime.Serialization.Formatters.Binary;
    using System.IO.Compression;
    using System.Drawing;

    internal enum FrameType
    {
        CSV,
        IMG
    }

    /// <summary>
    /// Dataset loader with utilities to split and shuffle.
    /// </summary>
    [Serializable]
    public class DataFrame
    {
        public DataFrame()
        {
            Data = new List<List<float>>();
            Columns = new List<string>();
            HotCode = new Dictionary<float, float>();
        }

        public int[] Shape
        {
            get
            {
                int[] result = new int[2];
                result[0] = Data.Count;
                result[1] = Data[0].Count;
                return result;
            }
        }

        /// <summary>
        /// Gets or sets the frame.
        /// </summary>
        /// <value>The frame.</value>
        public List<List<float>> Data { get; set; }

        private Dictionary<float, float> HotCode { get; set; }

        /// <summary>
        /// Gets or sets the columns.
        /// </summary>
        /// <value>
        /// The columns.
        /// </value>
        public List<string> Columns { get; set; }

        internal FrameType FrameType = FrameType.CSV;

        internal Tuple<int, int, int> imageDimension;

        /// <summary>
        /// Loads the dataset from CSV file.
        /// </summary>
        /// <param name="filePath">The CSV file path.</param>
        /// <param name="hasHeaders">If the CSV file have headers</param>
        /// <param name="seperatedByWhitespace">If row data is seperated by whitespace, otherwise comma by default</param>
        /// <param name="dataType">Data type of the CSV data.</param>
        public void LoadFromCsv(string filePath, bool hasHeaders = true, bool tabSeperator = false, Encoding encoding = null)
        {
            var lines = File.ReadAllLines(filePath);
            string seperator = tabSeperator ? "\t" : ",";
            int counter = 0;
            if (encoding == null)
                encoding = Encoding.Default;

            using (TextFieldParser parser = new TextFieldParser(filePath))
            {
                parser.TextFieldType = FieldType.Delimited;
                parser.SetDelimiters(seperator);
                parser.HasFieldsEnclosedInQuotes = true;
                if (parser.EndOfData)
                    throw new Exception("File is empty or not loaded properly.");

                if (hasHeaders)
                {
                    string[] fields = parser.ReadFields();
                    foreach (string field in fields)
                    {
                        Columns.Add(field);
                    }
                }

                List<float> record = new List<float>();
                float fieldValue = 0;
                while (!parser.EndOfData)
                {
                    string[] fields = parser.ReadFields();
                    record = new List<float>();
                    foreach (string field in fields)
                    {
                        float.TryParse(field, out fieldValue);
                        record.Add(fieldValue);
                    }

                    Data.Add(record);
                }

                if(Columns.Count == 0)
                {
                    for (int i = 0; i < Data[0].Count; i++)
                    {
                        Columns.Add(i.ToString());
                    }
                }
            }
        }

        /// <summary>
        /// Loads the image to a dataframe.
        /// </summary>
        /// <param name="imagePath">The image path.</param>
        /// <param name="resize">The resize value width X height. If pass null the image will be left it original size. Eg. Tuple.Create(32, 32)</param>
        /// <param name="grayScale">if set to <c>true</c> [gray scale].</param>
        public void LoadImage(string imagePath, Tuple<int, int> resize = null, bool grayScale = false)
        {
            Bitmap bmp = new Bitmap(Image.FromFile(imagePath));
            int channel = 3;
            List<float> bmpData = null;
            if (resize != null)
            {
                bmp = bmp.Resize(resize.Item1, resize.Item2, true);
            }

            if (grayScale)
            {
                bmp = bmp.ConvertGrayScale();
                bmpData = bmp.ParallelExtractGrayScale();
                channel = 1;
            }
            else
            {
                bmpData = bmp.ParallelExtractCHW();
            }

            Data.Add(bmpData);
            FrameType = FrameType.IMG;
            imageDimension = Tuple.Create(bmp.Width, bmp.Height, channel);
        }

        /// <summary>
        /// Splits the dataset to X and Y set.
        /// </summary>
        /// <param name="yCol">The y columns part of this split.</param>
        /// <param name="xCols">The x columns part of this split.</param>
        /// <returns>XY frame set.</returns>
        public XYFrame SplitXY(string yCol, params string[] xCols)
        {
            XYFrame result = new XYFrame();
            result.XFrame = new DataFrame();
            result.YFrame = new DataFrame();
            List<float> xFrameRow = new List<float>();
            List<float> yFrameRow = new List<float>();
            Data.ForEach((record) =>
            {
                xFrameRow.Clear();
                foreach (var col in Columns)
                {
                    if (xCols.Contains(col))
                    {
                        xFrameRow.Add(record[Columns.IndexOf(col)]);
                    }
                }

                yFrameRow.Add(record[Columns.IndexOf(yCol)]);

                result.XFrame.Add(xFrameRow);
                result.YFrame.Add(yFrameRow);
            });

            foreach (var col in Columns)
            {
                if (xCols.Contains(col))
                {
                    result.XFrame.Columns.Add(col);
                }
            }
            
            result.YFrame.Columns.Add(yCol);

            return result;
        }

        /// <summary>
        /// Splits the dataset to X and Y set.
        /// </summary>
        /// <param name="yCol">The y columns number.</param>
        /// <param name="xCols">The x columns numbers.</param>
        /// <returns>XY frame set.</returns>
        public XYFrame SplitXY(int yCol, int[] xCols)
        {
            XYFrame result = new XYFrame();
            result.XFrame = new DataFrame();
            result.YFrame = new DataFrame();
            List<float> xFrameRow = new List<float>();
            List<float> yFrameRow = new List<float>();

            if (xCols.Length != 2)
                throw new Exception("xCols length must be 2 with column range. Eg. [1, 10]");

            if (xCols[0] > xCols[1])
                throw new Exception("xCols range must have first value lower than second value.");

            Data.ForEach((record) =>
            {
                xFrameRow = new List<float>();
                yFrameRow = new List<float>();
                for (int i = xCols[0]; i <= xCols[1]; i++)
                {
                    xFrameRow.Add(record[i]);
                }
                
                yFrameRow.Add(record[yCol]);

                result.XFrame.Add(xFrameRow);
                result.YFrame.Add(yFrameRow);
            });

            if (result.XFrame.Columns.Count == 0)
            {
                for (int i = xCols[0]; i <= xCols[1]; i++)
                {
                    result.XFrame.Columns.Add(Columns[i]);
                }
            }
            
            if(result.YFrame.Columns.Count == 0)
                result.YFrame.Columns.Add(Columns[yCol]);

            return result;
        }

        /// <summary>
        /// Shuffles this dataset randomly.
        /// </summary>
        public void Shuffle()
        {
            List<List<float>> clone = Data;
            if (Data.Count > 0)
            {
                clone.Clear();
                Random random = new Random();

                while (Data.Count > 0)
                {
                    int row = random.Next(0, Data.Count);
                    clone.Add(Data[row]);
                    Data.RemoveAt(row);
                }
            }

            Data = clone;
        }

        /// <summary>
        /// Normalizes the dataframe with specified value.
        /// </summary>
        /// <param name="value">The value to normalize with.</param>
        public void Normalize(float? value = null)
        {
            List<List<float>> result = new List<List<float>>();

            if (value.HasValue)
            {
                Data.ForEach((record) =>
                {
                    List<float> r = new List<float>();
                    foreach (var item in record)
                    {
                        float recValue = item / value.Value;
                        r.Add(recValue);
                    }
                });
            }
            else
            {
                List<Tuple<float, float>> minmaxValues = new List<Tuple<float, float>>();
                for (int i = 0; i < Data[0].Count; i++)
                {
                    minmaxValues.Add(new Tuple<float, float>(Data.Min(x => (x[i])), Data.Max(x => (x[i]))));
                }

                Data.ForEach((record) =>
                {
                    List<float> r = new List<float>();
                    for (int i = 0; i < record.Count; i++)
                    {
                        var range = (minmaxValues[i].Item2 - minmaxValues[i].Item1);
                        if (range == 0)
                            range = float.Epsilon;

                        float normVal = (record[i] - minmaxValues[i].Item1) / range;
                        r.Add(normVal);
                    }

                    result.Add(r);
                });
            }

            Data = result;
        }

        public void MinMax(float min = 0, float max = 1)
        {
            if (min >= max)
                throw new ArgumentException("min value must is less than max");

            List<List<float>> result = new List<List<float>>();
            List<Tuple<float, float>> minmaxValues = new List<Tuple<float, float>>();
            for (int i = 0; i < Data[0].Count; i++)
            {
                minmaxValues.Add(new Tuple<float, float>(Data.Min(x => (x[i])), Data.Max(x => (x[i]))));
            }

            Data.ForEach((record) =>
            {
                List<float> r = new List<float>();
                for (int i = 0; i < record.Count; i++)
                {
                    float scale = (float)(max - min) / (float)(minmaxValues[i].Item2 - minmaxValues[i].Item1);
                    float offset = minmaxValues[i].Item1 * scale - min;
                    float normVal = (record[i] * scale - offset);
                    r.Add(normVal);
                }

                result.Add(r);
            });

            Data = result;
        }

        public void Add(List<float> data)
        {
            Add(data.ToArray());
        }

        public void Add(params float[] data)
        {
            if(Columns.Count == 0)
            {
                for(int i=0;i<data.Length;i++)
                {
                    Columns.Add(i.ToString());
                }
            }

            Data.Add(data.ToList());
        }

        /// <summary>
        /// Saves the data frame to a compressed stream.
        /// </summary>
        /// <param name="filepath">The filepath of the stream to save.</param>
        public void SaveStream(string filepath)
        {
            IFormatter formatter = new BinaryFormatter();
            using (var stream = new FileStream(filepath, FileMode.Create, FileAccess.Write, FileShare.None))
            {
                using (var gZipStream = new GZipStream(stream, CompressionMode.Compress))
                {
                    formatter.Serialize(gZipStream, this);
                }
            }
        }

        /// <summary>
        /// Loads the stream.
        /// </summary>
        /// <param name="filepath">The stream filepath to load the dataframe from.</param>
        /// <returns></returns>
        public static DataFrame LoadStream(string filepath)
        {
            if (!File.Exists(filepath))
                return null;

            IFormatter formatter = new BinaryFormatter();
            DataFrame frame = null;
            using (Stream stream = new FileStream(filepath, FileMode.Open, FileAccess.Read, FileShare.Read))
            {
                using (var gZipStream = new GZipStream(stream, CompressionMode.Decompress))
                {
                    frame = (DataFrame)formatter.Deserialize(gZipStream);
                }
            }

            return frame;
        }

        /// <summary>
        /// Called when [hot encode].
        /// </summary>
        public void OneHotEncode()
        {
            List<List<float>> encoded = new List<List<float>>();
            var group = Data.GroupBy(x => (x[0])).OrderBy(x=>(x.Key)).Select(x=>(x.Key)).ToList();
            HotCode = new Dictionary<float, float>();
            int counter = 0;
            Columns.Clear();
            foreach (var item in group)
            {
                HotCode.Add(counter, item);
                Columns.Add(counter.ToString());
                counter++;
            }

            foreach (var item in Data)
            {
                List<float> ylist = new List<float>();
                foreach (var h in HotCode)
                {
                    if(h.Value == item[0])
                    {
                        ylist.Add(1);
                        continue;
                    }

                    ylist.Add(0);
                }

                encoded.Add(ylist);
            }

            Data = encoded;
        }

        /// <summary>
        /// Get the dataframe for the columns specified
        /// </summary>
        /// <param name="columnNames">The column names.</param>
        /// <returns></returns>
        public DataFrame Partition(params string[] columnNames)
        {
            DataFrame result = new DataFrame();
            List<float> xFrameRow = new List<float>();
            Data.ForEach((record) =>
            {
                xFrameRow = new List<float>();
                foreach (var col in Columns)
                {
                    if (columnNames.Contains(col))
                    {
                        xFrameRow.Add(record[Columns.IndexOf(col)]);
                        result.Columns.Add(col);
                    }
                }

                result.Add(xFrameRow);
            });

            return result;
        }

        /// <summary>
        /// Get the dataframe for the columns specified
        /// </summary>
        /// <param name="columnNumbers">The column numbers.</param>
        /// <returns></returns>
        public DataFrame Partition(params int[] columnNumbers)
        {
            DataFrame result = new DataFrame();
            List<float> xFrameRow = new List<float>();
            Data.ForEach((record) =>
            {
                xFrameRow = new List<float>();
                for (int i = 0; i < Columns.Count; i++)
                {
                    if(columnNumbers.Contains(i))
                    {
                        xFrameRow.Add(record[i]);
                    }
                }

                result.Add(xFrameRow);
            });

            return result;
        }

        /// <summary>
        /// Converts the dataframe to time serires <see cref="XYFrame"/> dataframe.
        /// </summary>
        /// <param name="lookback">The lookback count.</param>
        /// <param name="useColumn">The column number to use for creating time series data.</param>
        /// <returns></returns>
        public XYFrame ConvertTimeSeries(int lookback = 1, int useColumn = 0)
        {
            XYFrame result = new XYFrame();
            List<float> data = Data.Select(x => (x[useColumn])).ToList();
            for (int i = 0; i < data.Count; i++)
            {
                if (i + lookback > data.Count - 1)
                    continue;

                List<float> values = new List<float>();
                int counter = i;
                while (counter < i + lookback)
                {
                    values.Add(data[counter]);
                    counter++;
                }

                result.XFrame.Add(values);
                result.YFrame.Add(new List<float>() { data[i + lookback] });
            }

            return result;
        }

        public void Reshape(params int[] shape)
        {
            Variable features = Variable.InputVariable(Shape, DataType.Float);
            Variable outfeatures = Variable.InputVariable(shape, DataType.Float);
            Function reshapeFunc = CNTKLib.Reshape(features, shape);
            List<float> vectorData = new List<float>();
            foreach (var item in Data)
            {
                vectorData.AddRange(item);
            }

            Value v = Value.CreateBatch<float>(Shape, vectorData, GlobalParameters.Device);
            var res = v.GetDenseData<float>(outfeatures);
            Data = new List<List<float>>();
            foreach (var item in res)
            {
                Data.Add(item.ToList());
            }
        }
    }
}
