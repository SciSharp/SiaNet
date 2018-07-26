using System.Collections.Generic;
using CNTK;

namespace SiaNet.Model.Data
{
    public interface IDataFrame
    {
        IEnumerable<float[]> Data { get; }
        int Length { get; }
        Shape DataShape { get; }
        void Reshape(Shape newShape);
        Value ToValue();
        float[] this[int index]
        {
            get;
            set;
        }
    }
}