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

    /// <summary>
    /// Dataset loader with utilities to split and shuffle.
    /// </summary>
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
        public void LoadFromCsv(string filePath, bool hasHeaders = true, bool seperatedByWhitespace = false, Encoding encoding = null)
        {
            var lines = File.ReadAllLines(filePath);
            string seperator = seperatedByWhitespace ? " " : ",";
            int counter = 0;
            if (encoding == null)
                encoding = Encoding.Default;

            CsvHelper.CsvReader reader = new CsvHelper.CsvReader(new StreamReader(filePath), new CsvHelper.Configuration.Configuration() { HasHeaderRecord = hasHeaders, Delimiter = seperator, Encoding = encoding });

            ////if (hasHeaders)
            ////{
            ////    reader.ReadHeader();
            ////    foreach (var item in columns)
            ////    {
            ////        Columns.Add(item);
            ////    }
            ////}
            ////else
            ////{
            ////    int columnCount = csvData.First().Count;
            ////    for (int i = 0; i < columnCount; i++)
            ////    {
            ////        Columns.Add(string.Format("Column_{0}", i));
            ////    }
            ////}

            ////foreach (var item in csvData)
            ////{
            ////    if (hasHeaders && counter == 0)
            ////    {
            ////        counter++;
            ////        continue;
            ////    }

            ////    try
            ////    {
            ////        var list = item.ToList<object>();
                    
            ////        row.ItemArray = item.ToArray<object>();
            ////        dt.Rows.Add(row);
            ////    }
            ////    catch
            ////    {
            ////    }

            ////    counter++;
            ////}

            ////Data = dt.AsDataView();
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
            Parallel.ForEach(Data, (record) =>
            {
                xFrameRow.Clear();
                foreach (var col in Columns)
                {
                    if (xCols.Contains(col))
                    {
                        xFrameRow.Add(record[Columns.IndexOf(col)]);
                        result.XFrame.Columns.Add(col);
                    }
                }

                yFrameRow.Add(record[Columns.IndexOf(yCol)]);

                result.XFrame.Add(xFrameRow);
                result.YFrame.Add(yFrameRow);
                result.YFrame.Columns.Add(yCol);
            });
            

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

            Parallel.ForEach(Data, (record) =>
            {
                xFrameRow.Clear();
                for (int i = xCols[0]; i <= xCols[1]; i++)
                {
                    xFrameRow.Add(record[i]);
                    result.XFrame.Columns.Add(Columns[i]);
                }

                yFrameRow.Add(record[yCol]);

                result.XFrame.Add(xFrameRow);
                result.YFrame.Add(yFrameRow);
                result.YFrame.Columns.Add(Columns[yCol]);
            });

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

            Parallel.ForEach(Data, (record) =>
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
    }
}
