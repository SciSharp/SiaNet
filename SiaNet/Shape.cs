using System;
using System.Linq;
using CNTK;

namespace SiaNet
{
    public class Shape
    {
        public Shape(params int[] values)
        {
            if (values.Any(i => i <= 0))
            {
                throw new ArgumentOutOfRangeException(nameof(values));
            }

            Dimensions = values;
        }

        public int[] Dimensions { get; set; }

        public int Length
        {
            get { return Dimensions.Aggregate(1, (current, dimension) => current * dimension); }
        }

        public int Rank
        {
            get => Dimensions.Length;
        }

        public static Shape FromNDShape(NDShape shape)
        {
            if (shape == null)
            {
                return null;
            }

            return new Shape(shape.Dimensions.ToArray());
        }

        public NDShape ToNDShape()
        {
            return NDShape.CreateNDShape(Dimensions);
        }
    }
}