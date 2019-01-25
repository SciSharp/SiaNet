using System;
using System.Collections.Generic;
using System.Text;
using TensorSharp.Expression;
using System.Linq;
using TensorSharp;

namespace SiaNet.Data
{
    public class DataFrame
    {
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
            UnderlayingTensor = TOps.NewContiguous(t);
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
                    frame.ToFrame(UnderlayingTensor.Narrow(1, start, count));

                return frame;
            }
        }

        public DataFrame this[uint index]
        {
            get
            {
                DataFrame frame = new DataFrame();
                var t = TOps.NewContiguous(UnderlayingTensor.Select(1, index));
                frame.ToFrame(t.Reshape(-1, 1));
                return frame;
            }
        }

        public void Norm(float value)
        {
            UnderlayingTensor = UnderlayingTensor.TVar().NormAll(value).Evaluate();
        }

        public Tensor GetBatch(int start, int size, int axis = 0)
        {
            if (start + size <= Shape[0])
            {
                return UnderlayingTensor.Narrow(axis, start, size);
            }
            else
            {
                return UnderlayingTensor.Narrow(axis, start, Shape[0] - start);
            }
        }

        public void Head(uint count = 5, string title = "")
        {
            UnderlayingTensor.Print(count, title);
        }
    }
}
