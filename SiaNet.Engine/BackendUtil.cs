using System;
using System.Collections.Generic;
using System.Text;
using System.Linq;

namespace SiaNet.Engine
{
    public class BackendUtil
    {
        public static IEnumerable<Tuple<int, int>> SwapsForReordering(int[] perm)
        {
            int j;
            for (int i = 0; i < perm.Length; ++i)
            {
                var p = perm[i];
                if (p != i && p != -1)
                {
                    j = i;
                    do
                    {
                        if (perm[j] < 0 || perm[j] >= perm.Length)
                            throw new InvalidOperationException("Invalid permutation");

                        yield return Tuple.Create(j, perm[j]);

                        var jOld = j;
                        j = perm[j];
                        perm[jOld] = -1;
                    } while (perm[j] != i);
                    perm[j] = j;
                }
            }
        }

        public static int[] CastShapeInt(long[] shape, bool reverse = false)
        {
            if (reverse)
                shape = shape.Reverse().ToArray();

            return Array.ConvertAll(shape, x => (int)x);
        }

        public static uint[] CastShapeUInt(long[] shape, bool reverse = false)
        {
            if (reverse)
                shape = shape.Reverse().ToArray();

            return Array.ConvertAll(shape, x => (uint)x);
        }
    }
}
