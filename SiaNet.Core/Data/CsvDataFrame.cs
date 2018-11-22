using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using CNTK;
using Accord.IO;

namespace SiaNet.Data
{
    public class CsvDataFrame : DataFrame
    {
        public List<string> Columns { get; set; }

        public void ReadCsv(string filePath, bool hasHeaders = false, string delimiter = ",")
        {
            List<List<float>> data = new List<List<float>>();
            using (CsvReader reader = new CsvReader(filePath, hasHeaders))
            {
                base.DataShape = new Shape(reader.FieldCount);

                var list = reader.ReadToEnd();
                Columns = reader.GetFieldHeaders().ToList();
                foreach (var item in list)
                {
                    var convertedData = item.ToList().ConvertAll(new Converter<string, float>((s) => {
                        float r = 0;
                        float.TryParse(s, out r);
                        return r;
                    })).ToArray();

                    base.Add(convertedData);
                }
            }
        }
        
        public IDataFrame this[params string[] columns]
        {
            get
            {
                DataFrame frame = new DataFrame(new Shape(columns.Length));
                
                foreach (string column in columns)
                {
                    if(!Columns.Contains(column))
                    {
                        throw new Exception(string.Format("Column: {0} not found", column));
                    }

                    int index = Columns.IndexOf(column);
                    
                    index = index * DataShape.TotalSize;
                    var data = new float[DataShape.TotalSize];
                    DataList.CopyTo(index, data, 0, data.Length);
                }

                return frame;
            }
        }
    }
}
