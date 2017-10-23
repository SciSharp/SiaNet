using CNTK;
using System;
using System.Collections.Generic;
using System.Data;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SiaNet.Model
{
    public class DataFrame
    {
        public DataView Frame { get; set; }

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

        public XYFrame SplitXY(string yCol, params string[] xCols)
        {
            XYFrame result = new XYFrame();
            result.XFrame.Frame = Frame.ToTable(false, xCols).AsDataView();
            result.YFrame.Frame = Frame.ToTable(false, yCol).AsDataView();

            return result;
        }

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
    }
}
