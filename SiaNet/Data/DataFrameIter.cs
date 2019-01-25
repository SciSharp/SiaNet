using System;
using System.Collections.Generic;
using System.Text;
using TensorSharp;

namespace SiaNet.Data
{
    public class DataFrameIter
    {
        private DataFrame frameX;
        private DataFrame frameY;

        private int batchSize { get; set; }

        private int current = 0;

        public long DataSize
        {
            get
            {
                return frameX.UnderlayingTensor.Shape[0];
            }
        }

        public DataFrameIter(DataFrame X, DataFrame Y = null)
        {
            frameX = X;
            frameY = Y;
            batchSize = 32;
            current = -batchSize;
        }

        public void SetBatchSize(int _batchSize)
        {
            batchSize = _batchSize;
            current = -batchSize;
        }

        public (Tensor, Tensor) GetBatch()
        {
            Tensor x = frameX.GetBatch(current, batchSize);
            Tensor y = frameY.GetBatch(current, batchSize);

            return (x, y);
        }

        public Tensor GetBatchX()
        {
            Tensor x = frameX.GetBatch(current, batchSize);
            return x;
        }

        public Tensor GetBatchY()
        {
            Tensor y = frameY.GetBatch(current, batchSize);
            return y;
        }

        public bool Next()
        {
            current += batchSize;
            return current < frameX.Shape[0];
        }

        public void Reset()
        {
            current = -batchSize;
        }
    }
}
