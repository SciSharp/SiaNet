using System;
using System.Linq;

namespace SiaNet
{
    public static class RandomGenerator
    {
        private static readonly Random random = new Random();

        public static double RandomDouble(double min, double max, int decimals = 5)
        {
            var precision = Math.Pow(10, decimals);

            return RandomInt((int) (min * precision), (int) (max * precision)) / precision;
        }


        public static int[] RandomIntArray(int count, int min, int max)
        {
            var d = new int[count];

            for (var i = 0; i < d.Length; i++)
            {
                d[i] = RandomInt(min, max);
            }

            return d;
        }

        public static int[] RandomIntArrayInclusive(int count, int min, int max)
        {
            var d = new int[count];

            for (var i = 0; i < d.Length; i++)
            {
                d[i] = RandomIntInclusive(min, max);
            }

            return d;
        }


        public static double[] RandomDoubleArray(int count, double min, double max, int decimals = 5)
        {
            var d = new double[count];

            for (var i = 0; i < d.Length; i++)
            {
                d[i] = RandomDouble(min, max, decimals);
            }

            return d;
        }

        public static double[] RandomDoubleArrayInclusive(int count, double min, double max, int decimals = 5)
        {
            var d = new double[count];

            for (var i = 0; i < d.Length; i++)
            {
                d[i] = RandomDoubleInclusive(min, max, decimals);
            }

            return d;
        }

        public static double RandomDoubleInclusive(double min, double max, int decimals = 5)
        {
            var precision = Math.Pow(10, decimals);

            return RandomInt((int) (min * precision), (int) (max * precision) + 1) / precision;
        }

        public static float[] RandomFloatArray(int count, float min, float max, int decimals = 5)
        {
            return RandomDoubleArray(count, min, max, decimals).Select(d => (float) d).ToArray();
        }

        public static int RandomInt(int min, int max)
        {
            return random.Next(min, max);
        }

        public static int RandomIntInclusive(int min, int max)
        {
            return RandomInt(min, max + 1);
        }
    }
}