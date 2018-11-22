using System;
using System.Linq;

namespace SiaNet.Data
{
    /// <summary>
    /// Data Frame List class which will hold the feature and label data used for training and validation
    /// </summary>
    /// <seealso cref="SiaNet.Model.Data.IDataFrameList" />
    public class DataFrameList<T> : IDataFrameList<T>
    {
        private readonly DataFrame<T> _features;
        private readonly DataFrame<T> _labels;

        /// <summary>
        /// Initializes a new instance of the <see cref="DataFrameList"/> class.
        /// </summary>
        /// <param name="featuresShape">The features shape.</param>
        /// <param name="labelsShape">The labels shape.</param>
        public DataFrameList(Shape featuresShape, Shape labelsShape) :
            this(new DataFrame<T>(featuresShape), new DataFrame<T>(labelsShape))
        {
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="DataFrameList"/> class.
        /// </summary>
        /// <param name="features">The features.</param>
        /// <param name="labels">The labels.</param>
        public DataFrameList(DataFrame<T> features, DataFrame<T> labels)
        {
            _features = features;
            _labels = labels;
        }

        /// <summary>
        /// Extracts the specified start.
        /// </summary>
        /// <param name="start">The start.</param>
        /// <param name="count">The count.</param>
        /// <returns></returns>
        /// <inheritdoc />
        public IDataFrameList<T> Extract(int start, int count)
        {
            var newList = new DataFrameList<T>(Features.DataShape, Labels.DataShape);

            for (var i = 0; i < count; i++)
            {
                var index = i + start;
                newList.AddFrame(Features[index], Labels[index]);
            }

            return newList;
        }


        /// <summary>
        /// Gets the features.
        /// </summary>
        /// <value>
        /// The features.
        /// </value>
        /// <inheritdoc />
        public virtual IDataFrame<T> Features
        {
            get => _features;
        }

        /// <summary>
        /// Gets the <see cref="Tuple{System.Single[], System.Single[]}"/> at the specified index.
        /// </summary>
        /// <value>
        /// The <see cref="Tuple{System.Single[], System.Single[]}"/>.
        /// </value>
        /// <param name="index">The index.</param>
        /// <returns></returns>
        /// <inheritdoc />
        public virtual Tuple<T[], T[]> this[int index]
        {
            get => new Tuple<T[], T[]>(Features[index], Labels[index]);
        }

        /// <summary>
        /// Gets the labels.
        /// </summary>
        /// <value>
        /// The labels.
        /// </value>
        /// <inheritdoc />
        public virtual IDataFrame<T> Labels
        {
            get => _labels;
        }

        /// <summary>
        /// Gets the length.
        /// </summary>
        /// <value>
        /// The length.
        /// </value>
        /// <inheritdoc />
        public virtual int Length
        {
            get => _features.Length;
        }

        /// <summary>
        /// Shuffles this instance.
        /// </summary>
        /// <inheritdoc />
        public virtual void Shuffle()
        {
            for (var i = Length - 1; i >= 0; i--)
            {
                var r = RandomGenerator.RandomInt(0, i);
                var features = _features[i];
                var label = _labels[i];
                _features[i] = _features[r];
                _labels[i] = _labels[r];
                _features[r] = features;
                _labels[r] = label;
            }
        }

        /// <summary>
        /// To the batch.
        /// </summary>
        /// <param name="batchId">The batch identifier.</param>
        /// <param name="batchSize">Size of the batch.</param>
        /// <returns></returns>
        /// <inheritdoc />
        public virtual IDataFrameList<T> ToBatch(int batchId, int batchSize)
        {
            var batchStart = batchId * batchSize;
            batchSize = Math.Min(batchSize, Length - batchStart);

            if (batchSize <= 0)
            {
                return null;
            }

            var newList = new DataFrameList<T>(Features.DataShape, Labels.DataShape);

            for (var i = 0; i < batchSize; i++)
            {
                var index = i + batchStart;
                newList.AddFrame(Features[index], Labels[index]);
            }

            return newList;
        }

        /// <summary>
        /// Add features and labels to the this frame.
        /// </summary>
        /// <param name="features">The features.</param>
        /// <param name="labels">The labels.</param>
        public virtual void AddFrame(T[] features, T[] labels)
        {
            _features.Add(features);
            _labels.Add(labels);
        }

        /// <summary>
        /// Add features and label to the this frame.
        /// </summary>
        /// <param name="features">The features.</param>
        /// <param name="label">The label.</param>
        public virtual void AddFrame(T[] features, T label)
        {
            _features.Add(features);
            _labels.Add(new[] { label });
        }
        
    }
}