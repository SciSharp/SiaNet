using System;
using System.Collections.Generic;
using System.Text;
using System.Linq;
using SiaNet.Engine;

namespace SiaNet.Data
{
    public class DataFrame
    {
        internal IBackend K = Global.CurrentBackend;

        internal Tensor UnderlayingTensor;

        public long[] Shape
        {
            get
            {
                return UnderlayingTensor.Shape;
            }
        }

        public void Reshape(params long[] newShape)
        {
            UnderlayingTensor = UnderlayingTensor.Reshape(Shape);
        }

        public Tensor GetTensor()
        {
            return UnderlayingTensor;
        }

        public virtual void ToFrame(Tensor t)
        {
            UnderlayingTensor = t;
        }

        public DataFrame this[uint start, uint end]
        {
            get
            {
                if (end <= start)
                {
                    throw new ArgumentException("End must be greater than start");
                }

                DataFrame frame = new DataFrame();
                var count = end - start + 1;
                if (count > 0)
                    frame.ToFrame(UnderlayingTensor.SliceCols(start, end));

                return frame;
            }
        }

        public DataFrame this[uint index]
        {
            get
            {
                DataFrame frame = new DataFrame();
                frame.ToFrame(UnderlayingTensor.SliceCols(index, index));

                return frame;
            }
        }

        public Tensor GetBatch(int start, int size, int axis = 0)
        {
            if (start + size <= Shape[0])
            {
                return UnderlayingTensor.SliceRows(start, start + size - 1);
            }
            else
            {
                return UnderlayingTensor.SliceRows(start, Shape[0] - 1);
            }
        }

        public void Head(uint count = 5, string title = "")
        {
            K.Print(UnderlayingTensor.SliceRows(0, count), title);
        }
    }
}
