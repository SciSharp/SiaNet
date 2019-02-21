using System;
using System.Collections.Generic;
using System.Text;
using System.Linq;
using System.IO;
using CsvHelper;
using System.Data;
using SiaNet.Engine;

namespace SiaNet.Data
{
    public class DataFrame2D : DataFrame
    {
        private long features;

        public DataFrame2D(long num_features)
            : base()
        {
            features = num_features;
        }

        public void Load(params float[] array)
        {
            long rows = array.LongLength / features;
            UnderlayingTensor = K.CreateVariable(array, new long[] { rows, features });
        }

        public override void ToFrame(Tensor t)
        {
            if(t.DimCount != 2)
            {
                throw new ArgumentException("2D tensor expected");
            }

            base.ToFrame(t);
        }

        public static DataFrame2D ReadCsv(string filepath, bool hasHeader = false)
        {
            List<float> allValues = new List<float>();
            DataFrame2D result = null;
            int columnCount = 0;
            using (TextReader fileReader = File.OpenText(filepath))
            {
                var csv = new CsvReader(fileReader);
                csv.Configuration.HasHeaderRecord = true;
                float value = 0;

                while (csv.Read())
                {
                    for (int i = 0; csv.TryGetField<float>(i, out value); i++)
                    {
                        allValues.Add(value);
                    }

                    if (columnCount == 0)
                        columnCount = allValues.Count;
                }

                result = new DataFrame2D(columnCount);
                result.Load(allValues.ToArray());
            }

            return result;
        }

        public static DataFrame2D FromDataTable(DataTable dt)
        {
            DataFrame2D result = null;

            
            return result;
        }
    }
}
