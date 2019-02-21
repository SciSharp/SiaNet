// This file was automatically generated using the AutoGenTool project
// If possible, edit the tool instead of editing this file directly

using System;
using System.Text;
using System.Numerics;
using System.Security;
using System.Runtime.InteropServices;

namespace SiaNet.Backend.ArrayFire.Interop
{
	internal static class af_config
	{
		internal const string dll = @"af";
	}

    [StructLayout(LayoutKind.Sequential)]
    public struct af_seq
    {
        public af_seq(double begin, double end, double step)
        {
            this.begin = begin;
            this.end = end;
            this.step = step;
        }

        // the C-API uses doubles so we have to do the same
        public double begin;
        public double end;
        public double step;

        public static readonly af_seq Span = new af_seq(1, 1, 0); // as defined in include/af/seq.h

        // implicit conversion from a number, only works in C# (not in F#)
        public static implicit operator af_seq(double begin) { return new af_seq(begin, begin, 1); }
    }
}
