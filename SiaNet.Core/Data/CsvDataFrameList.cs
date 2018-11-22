using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;

namespace SiaNet.Data
{
    public class CsvDataFrameList<T> : IDataFrameList<T>
    {
        protected readonly List<CsvDataFrameColumnSetting> ColumnSettings = new List<CsvDataFrameColumnSetting>();
        protected readonly string FileName;
        protected readonly bool HasHeader;
        protected readonly string LineSeparator;
        protected readonly List<Tuple<long, long>> LineTable = new List<Tuple<long, long>>();
        protected readonly bool ShouldTrim;
        protected readonly string ValueDelimiter;
        protected Encoding FileEncoding;

        public CsvDataFrameList(
            string fileName,
            Encoding encoding = null,
            string lineSeparator = "\r\n",
            string valueDelimiter = ",",
            bool shouldTrim = true,
            bool hasHeader = false)
        {
            FileName = fileName;
            FileEncoding = encoding == null ? Encoding.UTF8 : encoding;
            LineSeparator = lineSeparator;
            ValueDelimiter = valueDelimiter;
            ShouldTrim = shouldTrim;
            HasHeader = hasHeader;
            AnalyseFile();
        }

        private CsvDataFrameList(
            string fileName,
            Encoding encoding,
            string valueDelimiter,
            bool shouldTrim,
            Tuple<long, long>[] lineTable,
            CsvDataFrameColumnSetting[] columnSettings)
        {
            FileName = fileName;
            FileEncoding = encoding == null ? Encoding.UTF8 : encoding;
            ValueDelimiter = valueDelimiter;
            ShouldTrim = shouldTrim;
            LineTable.AddRange(lineTable);
            ColumnSettings.AddRange(columnSettings);
        }

        private CsvDataFrameColumnSetting[] Columns
        {
            get => ColumnSettings.ToArray();
        }

        public virtual int InitialSkip
        {
            get { return (int) Columns.Max(setting => setting.Delay + setting.Extra - 1); }
        }

        /// <inheritdoc />
        public IDataFrameList<T> Extract(int start, int count)
        {
            return new CsvDataFrameList<T>(FileName, FileEncoding, ValueDelimiter, ShouldTrim,
                LineTable.Skip(start).Take(count).ToArray(), Columns.Select(setting => setting.Clone()).ToArray());
        }

        /// <inheritdoc />
        public virtual IDataFrame<T> Features
        {
            get => ToBatch(0, LineTable.Count).Features;
        }

        /// <inheritdoc />
        public virtual Tuple<float[], float[]> this[int index]
        {
            get => ReadRecord(index);
        }

        /// <inheritdoc />
        public virtual IDataFrame<T> Labels
        {
            get => ToBatch(0, LineTable.Count).Labels;
        }


        /// <inheritdoc />
        public virtual int Length
        {
            get => LineTable.Count - InitialSkip;
        }


        /// <inheritdoc />
        public virtual void Shuffle()
        {
            for (var i = LineTable.Count - 1; i >= 0; i--)
            {
                var r = RandomGenerator.RandomInt(0, i);
                var position = LineTable[i];
                LineTable[i] = LineTable[r];
                LineTable[r] = position;
            }
        }

        /// <inheritdoc />
        public virtual IDataFrameList<T> ToBatch(int batchId, int batchSize)
        {
            var featureColumns = Columns.Where(setting => !setting.Ignore && !setting.IsLabel).DefaultIfEmpty()
                .Sum(setting => setting.Length);
            var labelColumns = Columns.Where(setting => !setting.Ignore && setting.IsLabel).DefaultIfEmpty()
                .Sum(setting => setting.Length);
            var lineIndex = batchId * batchSize;
            var newDataFrame = new DataFrameList<T>(featureColumns, labelColumns);

            foreach (var record in ReadRecords(lineIndex, batchSize))
            {
                newDataFrame.AddFrame(record.Item1, record.Item2);
            }

            return newDataFrame;
        }

        protected void AnalyseFile()
        {
            ColumnSettings.Clear();
            LineTable.Clear();
            var separatorIndex = 0;
            var lineSeparatorSize = FileEncoding.GetByteCount(LineSeparator);
            var lastLineStart = 0L;

            using (var file = new FileStream(FileName, FileMode.Open, FileAccess.Read))
            {
                // read BOM
                if (FileEncoding == null)
                {
                    var bom = new byte[4];
                    file.Read(bom, 0, 4);

                    if (bom[0] == 0x2b && bom[1] == 0x2f && bom[2] == 0x76)
                    {
                        FileEncoding = Encoding.UTF7;
                    }
                    else if (bom[0] == 0xef && bom[1] == 0xbb && bom[2] == 0xbf)
                    {
                        FileEncoding = Encoding.UTF8;
                    }
                    else if (bom[0] == 0xff && bom[1] == 0xfe) //UTF-16LE
                    {
                        FileEncoding = Encoding.Unicode;
                    }
                    else if (bom[0] == 0xfe && bom[1] == 0xff) //UTF-16BE
                    {
                        FileEncoding = Encoding.BigEndianUnicode;
                    }
                    else if (bom[0] == 0 && bom[1] == 0 && bom[2] == 0xfe && bom[3] == 0xff)
                    {
                        FileEncoding = Encoding.UTF32;
                    }
                    else
                    {
                        FileEncoding = Encoding.ASCII;
                    }
                }

                // Skip BOM
                if (FileEncoding.Equals(Encoding.UTF7) || FileEncoding.Equals(Encoding.UTF8))
                {
                    file.Seek(3, SeekOrigin.Begin);
                }
                else if (FileEncoding.Equals(Encoding.Unicode) || FileEncoding.Equals(Encoding.BigEndianUnicode))
                {
                    file.Seek(2, SeekOrigin.Begin);
                }
                else if (FileEncoding.Equals(Encoding.UTF32))
                {
                    file.Seek(4, SeekOrigin.Begin);
                }
                else
                {
                    file.Seek(0, SeekOrigin.Begin);
                }

                // Creating a line table
                var byteArray = new byte[FileEncoding.IsSingleByte ? 1 : 4];
                int read;

                while ((read = file.Read(byteArray, 0, byteArray.Length)) > 0)
                {
                    var charArray = FileEncoding.GetChars(byteArray, 0, read);
                    var charSize = FileEncoding.GetByteCount(new[] {charArray[0]});

                    if (byteArray.Length > charSize)
                    {
                        file.Position -= byteArray.Length - charSize;
                    }

                    if (charArray[0] == LineSeparator[separatorIndex])
                    {
                        separatorIndex++;

                        if (separatorIndex == LineSeparator.Length)
                        {
                            // found a line
                            LineTable.Add(new Tuple<long, long>(lastLineStart, file.Position - lineSeparatorSize));
                            lastLineStart = file.Position - lineSeparatorSize;
                            separatorIndex = 0;
                        }
                    }
                    else
                    {
                        separatorIndex = 0;
                    }
                }
            }

            var firstLine = ReadLine(0);

            if (HasHeader)
            {
                for (var i = 0; i < firstLine.Length; i++)
                {
                    ColumnSettings.Add(new CsvDataFrameColumnSetting(i, firstLine[i]));
                }

                LineTable.RemoveAt(0);
            }
            else
            {
                for (var i = 0; i < firstLine.Length; i++)
                {
                    ColumnSettings.Add(new CsvDataFrameColumnSetting(i));
                }
            }
        }

        protected virtual string[] ReadLine(int lineIndex)
        {
            return ReadLines(lineIndex, 1).FirstOrDefault();
        }

        protected virtual IEnumerable<string[]> ReadLines()
        {
            return ReadLines(0, LineTable.Count);
        }

        protected virtual IEnumerable<string[]> ReadLines(int lineIndex, int count)
        {
            using (var file = new FileStream(FileName, FileMode.Open, FileAccess.Read))
            {
                lineIndex = Math.Min(lineIndex - InitialSkip, 0);
                var lastLineIndex = Math.Min(LineTable.Count, lineIndex + count + InitialSkip) - 1;

                for (var i = lineIndex; i <= lastLineIndex - 1; i++)
                {
                    var lineStart = LineTable[i].Item1;
                    var lineEnd = LineTable[i].Item2;
                    var byteArray = new byte[lineEnd - lineStart];

                    if (file.Position != lineStart)
                    {
                        file.Seek(lineStart, SeekOrigin.Begin);
                    }

                    var read = file.Read(byteArray, 0, byteArray.Length);
                    var line = FileEncoding.GetString(byteArray, 0, read);

                    if (ShouldTrim)
                    {
                        yield return line.Trim().Split(new[] {ValueDelimiter}, StringSplitOptions.RemoveEmptyEntries)
                            .Select(s => s.Trim()).ToArray();
                    }

                    yield return line.Split(new[] {ValueDelimiter}, StringSplitOptions.RemoveEmptyEntries);
                }
            }
        }

        protected virtual Tuple<T[], T[]> ReadRecord(int lineIndex)
        {
            return ReadRecords(lineIndex, 1).FirstOrDefault();
        }

        protected virtual IEnumerable<Tuple<T[], T[]>> ReadRecords(int lineIndex, int count)
        {
            foreach (var column in Columns)
            {
                column.Reset();
            }

            var featureColumns = Columns.Where(setting => !setting.Ignore && !setting.IsLabel).ToArray();
            var labelColumns = Columns.Where(setting => !setting.Ignore && setting.IsLabel).ToArray();
            var columnsWithoutRange = Columns.Where(setting =>
                    !setting.Ignore && setting.Normalize && double.IsNaN(setting.ClipMax) ||
                    double.IsNaN(setting.ClipMin))
                .ToArray();

            if (columnsWithoutRange.Length > 0)
            {
                foreach (var recordInString in ReadLines())
                {
                    foreach (var column in columnsWithoutRange)
                    {
                        if (float.TryParse(recordInString[column.Index], out var value))
                        {
                            column.ClipMin = Math.Min(column.ClipMin, value);
                            column.ClipMax = Math.Max(column.ClipMax, value);
                        }
                    }
                }
            }

            foreach (var recordInString in ReadLines(lineIndex, count))
            {
                var features = new List<T>();
                var labels = new List<T>();

                for (var i = 0; i < featureColumns.Length; i++)
                {
                    var featureColumn = featureColumns[i];

                    if (!float.TryParse(recordInString[featureColumn.Index], out var value))
                    {
                        value = float.NaN;
                    }

                    var processedValue = featureColumn.Apply(value);

                    if (processedValue == null)
                    {
                        continue;
                    }

                    features.AddRange(processedValue);
                }

                for (var i = 0; i < labelColumns.Length; i++)
                {
                    var labelColumn = labelColumns[i];

                    if (!float.TryParse(recordInString[labelColumn.Index], out var value))
                    {
                        value = float.NaN;
                    }

                    var processedValue = labelColumn.Apply(value);

                    if (processedValue == null)
                    {
                        continue;
                    }

                    labels.AddRange(processedValue);
                }

                yield return new Tuple<float[], float[]>(features.ToArray(), labels.ToArray());
            }
        }
    }
}