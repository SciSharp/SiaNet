using System;

namespace SiaNet.Model.Data
{
    public interface IDataFrameList
    {
        IDataFrame Features { get; }

        Tuple<float[], float[]> this[int index] { get; }

        IDataFrame Labels { get; }
        int Length { get; }

        IDataFrameList Extract(int start, int count);
        void Shuffle();
        IDataFrameList ToBatch(int batchId, int batchSize);
    }
}