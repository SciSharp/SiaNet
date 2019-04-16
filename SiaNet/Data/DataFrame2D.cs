namespace SiaNet.Data
{
    using System;
    using System.Collections.Generic;
    using System.IO;
    using CsvHelper;
    using System.Data;
    using SiaNet.Engine;
    using NumSharp;

    /// <summary>
    /// 
    /// </summary>
    /// <seealso cref="SiaNet.Data.DataFrame" />
    public class DataFrame2D : DataFrame
    {
        private long features;

        /// <summary>
        /// Initializes a new instance of the <see cref="DataFrame2D"/> class.
        /// </summary>
        /// <param name="num_features">The number features this 2D frame have.</param>
        public DataFrame2D(long num_features)
            : base()
        {
            features = num_features;
        }

        public override void Load(params float[] data)
        {
            base.Load(data);
            Reshape(data.Length / features, features);
        }


        /// <summary>
        /// Reads the CSV file and load into the data frame.
        /// </summary>
        /// <param name="filepath">The filepath of the csv.</param>
        /// <param name="hasHeader">if set to <c>true</c> [will consider first row as column names].</param>
        /// <returns></returns>
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
    }
}
