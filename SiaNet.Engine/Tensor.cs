using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Runtime.Serialization;
using System.Text;
using System.Linq;

namespace SiaNet.Engine
{
    [DataContract]
    [DebuggerDisplay("{ToString()}")]
    public abstract class Tensor : IDisposable
    {
        public static IBackend K;

        public abstract string Name { get; set; }

        public Tensor()
        {

        }

        public Tensor(IBackend backend)
        {
            K = backend;
        }

        public long[] Shape
        {
            get { return K.GetShape(this); }
        }

        public int DimCount
        {
            get { return Shape.Length; }
        }

        public long ElementCount
        {
            get
            {
                if (Shape.Length > 0)
                    return Shape.Aggregate((a, x) => a * x);

                return 0;
            }
        }

        public DataType ElementType
        {
            get { return K.GetDataType(this); }
        }

        public bool IsVector
        {
            get
            {
                return DimCount == 1;
            }
        }

        public bool IsScalar
        {
            get
            {
                return ElementCount <= 1;
            }
        }

        public float ToScalar()
        {
            if(!IsScalar)
            {
                throw new Exception("Not scalar");
            }

            return ToArray().Cast<float>().FirstOrDefault();
        }

        public object Eval()
        {
            return K.Eval(this);
        }

        public float[] DataFloat
        {
            get
            {
                return ToArray().Cast<float>().ToArray();
            }
        }

        public TypeCode GetTypeCode()
        {
            return TypeCode.Object;
        }

        public Tensor Transpose()
        {
            return K.Transpose(this);
        }

        public Tensor Transpose(params int[] dims)
        {
            return K.Transpose(this, dims);
        }

        public Tensor Reshape(params long[] dims)
        {
            return K.Reshape(this, dims);
        }

        public Tensor SliceRows(long start, long end)
        {
            return K.SliceRows(this, start, end);
        }

        public Tensor SliceCols(long start, long end)
        {
            return K.SliceCols(this, start, end);
        }

        public Tensor RepeatTensor(long n, int dim)
        {
            return K.Tile(this, (int)n, dim);
        }

        public static Tensor operator *(float a, Tensor b)
        {
            return K.Mul(a, b);
        }

        public static Tensor operator *(Tensor a, float b)
        {
            return K.Mul(a, b);
        }

        public static Tensor operator *(Tensor a, Tensor b)
        {
            if (a.IsScalar)
            {
                return a.ToScalar() * b;
            }

            if (b.IsScalar)
            {
                return a * b.ToScalar();
            }

            var (lhs, rhs) = BroadcastTensor(a, b);
            return K.Mul(lhs, rhs);
        }

        public static Tensor operator /(float a, Tensor b)
        {
            return K.Div(a, b);
        }

        public static Tensor operator /(Tensor a, float b)
        {
            return K.Div(a, b);
        }

        public static Tensor operator /(Tensor a, Tensor b)
        {
            if (a.IsScalar)
            {
                return a.ToScalar() / b;
            }

            if (b.IsScalar)
            {
                return a / b.ToScalar();
            }

            var (lhs, rhs) = BroadcastTensor(a, b);
            return K.Div(lhs, rhs);
        }

        public static Tensor operator +(float a, Tensor b)
        {
            return K.Add(a, b);
        }

        public static Tensor operator +(Tensor a, Tensor b)
        {
            if (a.IsScalar)
            {
                return a.ToScalar() + b;
            }

            if (b.IsScalar)
            {
                return a + b.ToScalar();
            }

            var (lhs, rhs) = BroadcastTensor(a, b);
            return K.Add(lhs, rhs);
        }

        public static Tensor operator +(Tensor a, float b)
        {
            return K.Add(a, b);
        }

        public static Tensor operator -(float a, Tensor b)
        {
            return K.Sub(a, b);
        }

        public static Tensor operator -(Tensor a, Tensor b)
        {
            if (a.IsScalar)
            {
                return a.ToScalar() - b;
            }

            if (b.IsScalar)
            {
                return a - b.ToScalar();
            }

            var (lhs, rhs) = BroadcastTensor(a, b);
            return K.Sub(lhs, rhs);
        }

        public static Tensor operator -(Tensor a, float b)
        {
            return K.Sub(a, b);
        }

        private static ValueTuple<Tensor, Tensor> BroadcastTensor(Tensor lhs, Tensor rhs)
        {
            if (lhs.DimCount == rhs.DimCount && !lhs.IsVector && !lhs.IsVector)
            {
                if (lhs.Shape[0] == rhs.Shape[0] && (lhs.Shape[1] == 1 || rhs.Shape[1] == 1))
                {
                    if (lhs.Shape[1] == 1)
                    {
                        lhs = lhs.RepeatTensor(rhs.Shape[1], 0);
                    }

                    if (rhs.Shape[1] == 1)
                    {
                        rhs = rhs.RepeatTensor(lhs.Shape[1], 0);
                    }
                }

                if (lhs.Shape[1] == rhs.Shape[1] && (lhs.Shape[0] == 1 || rhs.Shape[0] == 1))
                {
                    if (lhs.Shape[0] == 1)
                    {
                        lhs = lhs.RepeatTensor(rhs.Shape[0], 1);
                    }

                    if (rhs.Shape[0] == 1)
                    {
                        rhs = rhs.RepeatTensor(lhs.Shape[0], 1);
                    }
                }

                if (lhs.Shape[1] == 1 && rhs.Shape[0] == 1)
                {
                    if (lhs.Shape[1] == 1)
                    {
                        lhs = lhs.RepeatTensor(rhs.Shape[1], 0);
                    }

                    if (rhs.Shape[0] == 1)
                    {
                        rhs = rhs.RepeatTensor(lhs.Shape[0], 1);
                    }
                }

                if (lhs.Shape[0] == 1 || rhs.Shape[1] == 1)
                {
                    if (lhs.Shape[0] == 1)
                    {
                        lhs = lhs.RepeatTensor(rhs.Shape[0], 1);
                    }

                    if (rhs.Shape[1] == 1)
                    {
                        rhs = rhs.RepeatTensor(lhs.Shape[1], 0);
                    }
                }
            }
            else if(lhs.IsVector && !rhs.IsVector)
            {
                lhs = lhs.RepeatTensor(rhs.Shape[1], 0).Reshape(rhs.Shape[1], -1);
            }
            else if (rhs.IsVector && !lhs.IsVector)
            {
                rhs = rhs.RepeatTensor(lhs.Shape[1], 0).Reshape(lhs.Shape[1], -1); ;
            }

            return (lhs, rhs);
        }

        public static Tensor operator >(Tensor a, Tensor b)
        {
            return K.GreaterThan(a, b);
        }

        public static Tensor operator >=(Tensor a, Tensor b)
        {
            return K.GreaterThanEqual(a, b);
        }

        public static Tensor operator <(Tensor a, Tensor b)
        {
            return K.LessThan(a, b);
        }

        public static Tensor operator <=(Tensor a, Tensor b)
        {
            return K.LessThanEqual(a, b);
        }

        public static Tensor operator >(float a, Tensor b)
        {
            return K.GreaterThan(a, b);
        }

        public static Tensor operator >=(float a, Tensor b)
        {
            return K.GreaterThanEqual(a, b);
        }

        public static Tensor operator <(float a, Tensor b)
        {
            return K.LessThan(a, b);
        }

        public static Tensor operator <=(float a, Tensor b)
        {
            return K.LessThanEqual(a, b);
        }

        public static Tensor operator >(Tensor a, float b)
        {
            return K.GreaterThan(a, b);
        }

        public static Tensor operator >=(Tensor a, float b)
        {
            return K.GreaterThanEqual(a, b);
        }

        public static Tensor operator <(Tensor a, float b)
        {
            return K.LessThan(a, b);
        }

        public static Tensor operator <=(Tensor a, float b)
        {
            return K.LessThanEqual(a, b);
        }

        public ValueTuple<long, long, long, long, long> GetConv3DShape()
        {
            return (Shape[0], Shape[1], Shape[2], Shape[3], Shape[4]);
        }

        public ValueTuple<long, long, long, long> GetConv2DShape()
        {
            return (Shape[0], Shape[1], Shape[2], Shape[3]);
        }

        public ValueTuple<long, long, long> GetConv1DShape()
        {
            return (Shape[0], Shape[1], Shape[2]);
        }

        public Array ToArray()
        {
            return K.GetArray(this);
        }

        public void Print(string title = "")
        {
            K.Print(this, title);
        }

        public void Dispose()
        {
            K.Dispose(this);
        }
    }
}
