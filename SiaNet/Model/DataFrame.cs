namespace SiaNet.Model
{
    using CNTK;
    using System;
    using System.Collections.Generic;
    using System.Data;
    using System.IO;
    using System.Linq;
    using System.Text;
    using System.Threading.Tasks;
    using Microsoft.VisualBasic.FileIO;
    using System.Runtime.Serialization;
    using System.Runtime.Serialization.Formatters.Binary;
    using System.IO.Compression;

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
        }

        /// <summary>
        /// Gets or sets the frame.
        /// </summary>
        /// <value>The frame.</value>
        public List<List<float>> Data { get; set; }

        /// <summary>
        /// Gets or sets the columns.
        /// </summary>
        /// <value>
        /// The columns.
        /// </value>
        public List<string> Columns { get; set; }

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

            for (int i = xCols[0]; i <= xCols[1]; i++)
            {
                result.XFrame.Columns.Add(Columns[i]);
            }
            
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

        public void Normalize(float value)
        {
            List<List<float>> result = new List<List<float>>();

            Data.ForEach((record) =>
            {
                List<float> r = new List<float>();
                foreach (var item in record)
                {
                    float recValue = item / value;
                    r.Add(recValue);
                }
            });
        }

        public void Add(List<float> data)
        {
            Data.Add(data);
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
    }
}
