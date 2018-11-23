using System;

namespace SiaNet.Data
{
    public interface IDataFrameList<T>
    {
        IDataFrame<T> Features { get; }

        Tuple<T[], T[]> this[int index] { get; }

        IDataFrame<T> Labels { get; }
        int Length { get; }

        IDataFrameList<T> Extract(int start, int count);
        void Shuffle();
        IDataFrameList<T> ToBatch(int batchId, int batchSize);
    }
}