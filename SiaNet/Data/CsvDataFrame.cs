using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using CNTK;
using Accord.IO;

namespace SiaNet.Data
{
    public class CsvDataFrame<T> : DataFrame<T>
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
                    var convertedData = item.ToList().ConvertAll(new Converter<string, T>((s) => {
                        T r = default(T);
                        try
                        {
                            r = (T)Convert.ChangeType(s, typeof(T));
                        }
                        catch { }
                        return r;
                    })).ToArray();

                    base.Add(convertedData);
                }
            }
        }
        
        public DataFrame<T> this[params string[] columns]
        {
            get
            {
                DataFrame<T> frame = new DataFrame<T>(new Shape(columns.Length));
                
                foreach (string column in columns)
                {
                    if(!Columns.Contains(column))
                    {
                        throw new Exception(string.Format("Column: {0} not found", column));
                    }

                    int index = Columns.IndexOf(column);
                    
                    index = index * DataShape.TotalSize;
                    var data = new T[DataShape.TotalSize];
                    frame.Add(data);
                }

                return frame;
            }
        }

        public DataFrame<T> this[int start, int count]
        {
            get
            {
                DataFrame<T> frame = new DataFrame<T>(new Shape(count));

                int i = start;
                while(i < DataList.Count)
                {
                    T[] data = new T[count];
                    base.DataList.CopyTo(i, data, 0, count);
                    frame.Add(data);
                    i += base.DataShape.TotalSize;
                }

                return frame;
            }
        }
    }
}
