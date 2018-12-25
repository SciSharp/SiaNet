using System;
using System.Collections.Generic;
using System.Text;
using TensorSharp;

namespace SiaNet.Data
{
    public interface IFrameIter
    {
        void SetBatchSize(int batchSize);
        (Tensor, Tensor) GetBatch();
        bool Next();
        void Reset();
    }
}
