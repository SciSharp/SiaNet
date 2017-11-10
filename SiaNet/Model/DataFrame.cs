namespace SiaNet.Model
{
    using CNTK;
    using System;
    using System.Collections.Generic;
    using System.Data;
    using System.IO;
    using System.Linq;

    /// <summary>
    /// Dataset loader with utilities to split and shuffle
    /// </summary>
    public class DataFrame
    {
        /// <summary>
        /// Gets or sets the frame.
        /// </summary>
        /// <value>The frame.</value>
        public DataView Frame { get; set; }

        /// <summary>
        /// Loads the dataset from CSV file.
        /// </summary>
        /// <param name="filePath">The CSV file path.</param>
        /// <param name="hasHeaders">If the CSV file have headers</param>
        /// <param name="seperatedByWhitespace">If row data is seperated by whitespace, otherwise comma by default</param>
        /// <param name="dataType">Data type of the CSV data.</param>
        public void LoadFromCsv(string filePath, bool hasHeaders = true, bool seperatedByWhitespace = false, DataType dataType = DataType.Float)
        {
            var lines = File.ReadAllLines(filePath);
            char seperator = seperatedByWhitespace ? ' ' : ',';
            var csvData = from line in lines select (line.Split(seperator)).ToArray().ToList();
            int counter = 0;
            
            DataTable dt = new DataTable();
            if (hasHeaders)
            {
                var columns = csvData.First();
                foreach (var item in columns)
                {
                    dt.Columns.Add(new DataColumn(item, GetDatatableDType(dataType)));
                }
            }
            else
            {
                int columnCount = csvData.First().Count;
                for (int i = 0; i < columnCount; i++)
                {
                    dt.Columns.Add(new DataColumn(string.Format("Column_{0}", i), GetDatatableDType(dataType)));
                }
            }

            foreach (var item in csvData)
            {
                if (hasHeaders && counter == 0)
                {
                    counter++;
                    continue;
                }

                try
                {
                    DataRow row = dt.NewRow();
                    var list = item.ToList<object>();
                    
                    row.ItemArray = item.ToArray<object>();
                    dt.Rows.Add(row);
                }
                catch
                {
                }

                counter++;
            }

            Frame = dt.AsDataView();
        }

        /// <summary>
        /// Gets the type of the datatable d.
        /// </summary>
        /// <param name="dtype">The dtype.</param>
        /// <returns>Type.</returns>
        private Type GetDatatableDType(DataType dtype)
        {
            Type t = typeof(float);

            switch (dtype)
            {
                case DataType.Double:
                    t = typeof(double);
                    break;
                case DataType.UChar:
                    t = typeof(string);
                    break;
                default:
                    break;
            }

            return t;
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
            result.XFrame.Frame = Frame.ToTable(false, xCols).AsDataView();
            result.YFrame.Frame = Frame.ToTable(false, yCol).AsDataView();

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
            DataTable dt = Frame.ToTable();
            List<string> xColumns = new List<string>();
            string yColumn = dt.Columns[yCol].ColumnName;

            for (int i = xCols[0]; i < xCols[1]; i++)
            {
                xColumns.Add(dt.Columns[i].ColumnName);
            }


            result.XFrame.Frame = Frame.ToTable(false, xColumns.ToArray()).AsDataView();
            result.YFrame.Frame = Frame.ToTable(false, yColumn).AsDataView();

            return result;
        }

        /// <summary>
        /// Shuffles this dataset randomly.
        /// </summary>
        public void Shuffle()
        {
            DataTable table = Frame.ToTable();
            DataTable clone = table.Clone();
            if (table.Rows.Count > 0)
            {
                clone.Clear();
                Random random = new Random();

                while (table.Rows.Count > 0)
                {
                    int row = random.Next(0, table.Rows.Count);
                    clone.ImportRow(table.Rows[row]);
                    table.Rows[row].Delete();
                }
            }

            Frame = clone.AsDataView();
        }

        public void Normalize(double value)
        {
            DataTable dt = Frame.ToTable();
        }
    }
}
