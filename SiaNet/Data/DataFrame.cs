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
        internal Tensor underlayingVariable;

        public long[] Shape
        {
            get
            {
                return underlayingVariable.Shape;
            }
        }

        public void Reshape(params long[] newShape)
        {
            underlayingVariable = underlayingVariable.View(Shape);
        }

        public Tensor GetTensor()
        {
            return underlayingVariable;
        }

        public virtual void ToFrame(Tensor t)
        {
            underlayingVariable = t;
        }

        public DataFrame this[int start, int end]
        {
            get
            {
                if (start < 1)
                {
                    throw new ArgumentException("Start cannot be less than 1");
                }

                if (end < start)
                {
                    throw new ArgumentException("End must be greater than start");
                }

                start = start - 1;
                DataFrame frame = new DataFrame();
                frame.ToFrame(underlayingVariable.Narrow(1, start, end - start));

                return frame;
            }
        }

        public void Norm(float value)
        {
            underlayingVariable = underlayingVariable.TVar().NormAll(value).Evaluate();
        }

        public Tensor GetBatch(int start, int size, int axis = 0)
        {
            if (start + size <= Shape[0])
            {
                return underlayingVariable.Narrow(axis, start, size);
            }
            else
            {
                return underlayingVariable.Narrow(axis, start, Shape[0] - start);
            }
        }

        public void Print()
        {
            underlayingVariable.Print();
        }
    }
}
