using System.Collections.Generic;
using CNTK;

namespace SiaNet.Data
{
    public interface IDataFrame<T>
    {
        IEnumerable<T[]> Data { get; }
        int Length { get; }
        Shape DataShape { get; }
        void Reshape(Shape newShape);
        Value ToValue();
        T[] this[int index]
        {
            get;
            set;
        }
    }
}