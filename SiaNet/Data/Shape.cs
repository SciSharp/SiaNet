using System;
using System.Linq;
using CNTK;

namespace SiaNet.Data
{
    public class Shape
    {
        public Shape(params int[] values)
        {
            //if (values.Any(i => i <= 0))
            //{
            //    throw new ArgumentOutOfRangeException(nameof(values));
            //}

            Dimensions = values;
        }

        public int[] Dimensions { get; set; }

        public int Rank
        {
            get => Dimensions.Length;
        }

        public int TotalSize
        {
            get { return Dimensions.Aggregate(1, (current, dimension) => current * dimension); }
        }

        public int this[int index]
        {
            get => Dimensions[index];
        }

        public static implicit operator NDShape(Shape s)
        {
            return NDShape.CreateNDShape(s.Dimensions);
        }

        public static implicit operator Shape(int i)
        {
            return new Shape(i);
        }

        public static implicit operator Shape(int[] a)
        {
            return new Shape(a);
        }

        public static implicit operator Shape(Tuple<int, int> t)
        {
            return new Shape(t.Item1, t.Item2);
        }

        public static implicit operator Shape(Tuple<int, int, int> t)
        {
            return new Shape(t.Item1, t.Item2, t.Item3);
        }

        public static implicit operator Shape(Tuple<int, int, int, int> t)
        {
            return new Shape(t.Item1, t.Item2, t.Item3, t.Item4);
        }

        public static implicit operator Shape(Tuple<int, int, int, int, int> t)
        {
            return new Shape(t.Item1, t.Item2, t.Item3, t.Item4, t.Item5);
        }

        public static implicit operator Shape(NDShape s)
        {
            if (s == null)
            {
                return null;
            }

            return new Shape(s.Dimensions.ToArray());
        }
    }
}