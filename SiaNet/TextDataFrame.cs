using CNTK;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SiaNet
{
    /// <summary>
    /// Wrapper for the data frame to load text data and generate dataframe. Option to add the data manually or load from folder having stored text as files.
    /// </summary>
    /// <seealso cref="SiaNet.Model.XYFrame" />
    public class TextDataFrame : XYFrame
    {
        private int features = 0;
        private int labels = 0;
        private bool padSequence = true;
        private string folder;
        private bool fromFolder;

        /// <summary>
        /// Initializes a new instance of the <see cref="TextDataFrame"/> class.
        /// </summary>
        /// <param name="maxWords">The maximum words.</param>
        /// <param name="classes">The number of label classes.</param>
        /// <param name="padSequence">Pad the word input with zeros in case less than max words otherwise truncate.</param>
        public TextDataFrame(int maxWords, int classes, bool padSequence = true)
            : base()
        {
            this.features = maxWords;
            this.labels = classes;
            this.padSequence = padSequence;
        }

        /// <summary>
        /// Build the dataset by adding features and label
        /// </summary>
        /// <param name="features">The features.</param>
        /// <param name="label">The label.</param>
        public void Add(List<float> features, float label)
        {
            if (padSequence)
                features = PadData(features);

            XFrame.Add(features);
            List<float> ydatalist = new List<float>();
            for (int i = 0; i < labels; i++)
            {
                if (i == label)
                {
                    ydatalist.Add(1);
                    continue;
                }

                ydatalist.Add(0);
            }

            YFrame.Add(ydatalist);
        }

        /// <summary>
        /// Invoke this method to pads the sequence with max words specified in the constructor
        /// </summary>
        public void PadSequence()
        {
            List<List<float>> paddedData = new List<List<float>>();
            foreach (var item in XFrame.Data)
            {
                paddedData.Add(PadData(item));
            }

            XFrame.Data = paddedData;
        }

        /// <summary>
        /// Pads the data.
        /// </summary>
        /// <param name="input">The input.</param>
        /// <returns></returns>
        private List<float> PadData(List<float> input)
        {
            List<float> result = new List<float>();

            for (int i = 0; i < features; i++)
            {
                if (input.Count < i)
                    result.Add(input[i]);
                else
                    result.Add(0);
            }

            return result;
        }
    }
}
