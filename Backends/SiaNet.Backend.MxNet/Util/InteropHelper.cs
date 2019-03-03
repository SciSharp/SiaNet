using System;

// ReSharper disable once CheckNamespace
namespace SiaNet.Backend.MxNetLib
{

    internal static class InteropHelper
    {

        #region Methods

        public static IntPtr[] ToPointerArray(IntPtr ptr, uint count)
        {
            unsafe
            {
                var array = new IntPtr[count];
                var p = (void**)ptr;
                for (var i = 0; i < count; i++)
                    array[i] = (IntPtr)p[i];

                return array;
            }
        }

        public static float[] ToFloatArray(IntPtr ptr, uint count)
        {
            unsafe
            {
                var array = new float[count];
                var p = (float*)ptr;
                for (var i = 0; i < count; i++)
                    array[i] = p[i];

                return array;
            }
        }

        public static uint[] ToUInt32Array(IntPtr ptr, uint count)
        {
            unsafe
            {
                var array = new uint[count];
                var p = (uint*)ptr;
                for (var i = 0; i < count; i++)
                    array[i] = p[i];

                return array;
            }
        }

        public static ulong[] ToUInt64Array(IntPtr ptr, uint count)
        {
            unsafe
            {
                var array = new ulong[count];
                var p = (ulong*)ptr;
                for (var i = 0; i < count; i++)
                    array[i] = p[i];

                return array;
            }
        }

        #endregion

    }

}
