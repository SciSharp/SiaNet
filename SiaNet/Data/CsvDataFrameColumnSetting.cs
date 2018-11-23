using System;
using System.Collections.Generic;
using System.Linq;

namespace SiaNet.Data
{
    public class CsvDataFrameColumnSetting
    {
        private readonly List<float> _history = new List<float>();

        public CsvDataFrameColumnSetting(int index, string name = null)
        {
            Index = index;
            Name = !string.IsNullOrWhiteSpace(name) ? name : index.ToString("D3");
        }

        public double ClipMax { get; set; } = double.NaN;
        public double ClipMin { get; set; } = double.NaN;
        public int Decimals { get; set; } = int.MaxValue;
        public uint Delay { get; set; }

        public uint Extra { get; set; }
        public bool Ignore { get; set; }
        public int Index { get; }
        public bool IsLabel { get; set; }

        public int Length
        {
            get => (int) (Extra + 1);
        }

        public string Name { get; }
        public bool Normalize { get; set; }
        public double NormalizeMax { get; set; } = 1d;
        public double NormalizeMin { get; set; } = -1d;
        public bool Relative { get; set; }
        public bool SkipNotNumeric { get; set; } = true;

        public virtual float[] Apply(float f)
        {
            if (float.IsNaN(f) && SkipNotNumeric)
            {
                return null;
            }

            var d = (double) f;
            var lastValue = _history.Cast<float?>().LastOrDefault();

            if (Relative)
            {
                if (lastValue != null && !float.IsNaN(lastValue.Value))
                {
                    d = d - lastValue.Value;
                }
                else
                {
                    AddHistory((float) d);

                    return null;
                }
            }

            if (Normalize &&
                !double.IsNaN(ClipMax) &&
                !double.IsNaN(ClipMin) &&
                !double.IsNaN(NormalizeMax) &&
                !double.IsNaN(NormalizeMin))
            {
                d = Math.Max(Math.Min(d, ClipMax), ClipMin);
                var dataRange = ClipMax - ClipMin;
                var normalizedRange = NormalizeMax - NormalizeMin;
                d = (d - ClipMin) / dataRange * normalizedRange + NormalizeMin;
            }

            AddHistory((float) Math.Round(d, Decimals));

            return GetHistory();
        }

        public virtual CsvDataFrameColumnSetting Clone()
        {
            return new CsvDataFrameColumnSetting(Index, Name)
            {
                ClipMin = ClipMin,
                NormalizeMin = NormalizeMin,
                SkipNotNumeric = SkipNotNumeric,
                ClipMax = ClipMax,
                IsLabel = IsLabel,
                NormalizeMax = NormalizeMax,
                Relative = Relative,
                Decimals = Decimals,
                Normalize = Normalize,
                Ignore = Ignore,
                Extra = Extra,
                Delay = Delay
            };
        }

        public virtual void Reset()
        {
            _history.Clear();
        }

        private void AddHistory(float f)
        {
            while (_history.Count >= Delay + Length)
            {
                _history.RemoveAt(0);
            }

            _history.Add(f);
        }

        private float[] GetHistory()
        {
            if (_history.Count != Delay + Length)
            {
                return null;
            }

            return _history.ToArray().Reverse().Skip((int) Delay).Take(Length).ToArray();
        }
    }
}