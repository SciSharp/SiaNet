using System;

namespace SiaNet.Model.Data
{
    public class DataFrameList : IDataFrameList
    {
        private readonly DataFrame _features;
        private readonly DataFrame _labels;

        public DataFrameList(Shape featuresShape, Shape labelsShape) :
            this(new DataFrame(featuresShape), new DataFrame(labelsShape))
        {
        }

        public DataFrameList(DataFrame features, DataFrame labels)
        {
            _features = features;
            _labels = labels;
        }

        /// <inheritdoc />
        public IDataFrameList Extract(int start, int count)
        {
            var newList = new DataFrameList(Features.DataShape, Labels.DataShape);

            for (var i = 0; i < count; i++)
            {
                var index = i + start;
                newList.AddFrame(Features[index], Labels[index]);
            }

            return newList;
        }


        /// <inheritdoc />
        public virtual IDataFrame Features
        {
            get => _features;
        }

        /// <inheritdoc />
        public virtual Tuple<float[], float[]> this[int index]
        {
            get => new Tuple<float[], float[]>(Features[index], Labels[index]);
        }

        /// <inheritdoc />
        public virtual IDataFrame Labels
        {
            get => _labels;
        }

        /// <inheritdoc />
        public virtual int Length
        {
            get => _features.Length;
        }

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

        /// <inheritdoc />
        public virtual IDataFrameList ToBatch(int batchId, int batchSize)
        {
            var batchStart = batchId * batchSize;
            batchSize = Math.Min(batchSize, Length - batchStart);

            if (batchSize <= 0)
            {
                return null;
            }

            var newList = new DataFrameList(Features.DataShape, Labels.DataShape);

            for (var i = 0; i < batchSize; i++)
            {
                var index = i + batchStart;
                newList.AddFrame(Features[index], Labels[index]);
            }

            return newList;
        }

        public virtual void AddFrame(float[] features, float[] labels)
        {
            _features.Add(features);
            _labels.Add(labels);
        }

        public virtual void AddFrame(float[] features, float label)
        {
            _features.Add(features);
            _labels.Add(new[] {label});
        }
    }
}