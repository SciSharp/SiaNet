using CsvHelper;
using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using System.Linq;
using SiaNet.Backend;

namespace SiaNet.Data
{
    public class CsvDataFrame : DataFrame
    {
        public List<string> Columns { get; set; }

        private string _path = string.Empty;

        private bool _hasHeaders;

        private string _delimiter;

        private Encoding _encoding;

        public CsvDataFrame(string path , bool hasHeaders = false, string delimiter = ",", Encoding encoding = null)
        {
            _path = path;
            _hasHeaders = hasHeaders;
            _delimiter = delimiter;
            _encoding = encoding;
        }

        public void ReadCsv()
        {
            List<float> data = new List<float>();
            using (TextReader fileReader = File.OpenText(_path))
            {
                var csv = new CsvReader(fileReader);
                csv.Configuration.Delimiter = _delimiter;
                csv.Configuration.HasHeaderRecord = _hasHeaders;

                if (_encoding != null)
                    csv.Configuration.Encoding = _encoding;

                if(_hasHeaders)
                {
                    string[] columnNames = csv.Parser.Read();
                    _cols = (uint)columnNames.Length;
                    Columns = columnNames.ToList();
                }

                while (csv.Read())
                {
                    string[] rowData = csv.Parser.Context.Record;
                    if (_cols == 0)
                    {
                        _cols = (uint)rowData.Length;
                    }

                    foreach (string item in rowData)
                    {
                        DataList.Add(float.Parse(item));
                    }
                }
            }

            GenerateVariable();
        }
    }
}
