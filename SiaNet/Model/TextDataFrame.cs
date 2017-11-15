using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SiaNet.Model
{
    public class TextDataFrame : DataFrame
    {
        private int[] features;
        private int labels;
        private string folder;
        private bool fromFolder;
    }

    public class TextDataItem
    {
        public TextDataItem()
        {
            WordSequence = new List<KeyValuePair<string, float>>();
        }

        public string Text { get; set; }

        public int Label { get; set; }

        public List<KeyValuePair<string, float>> WordSequence { get; set; }

        internal void PadSequence(int maxWords)
        {
            List<KeyValuePair<string, float>> result = new List<KeyValuePair<string, float>>();
            for (int i = 0; i < maxWords; i++)
            {
                if (WordSequence.Count < i)
                    result.Add(WordSequence[i]);
                else
                    result.Add(new KeyValuePair<string, float>("", 0));
            }
        }
    }
}
