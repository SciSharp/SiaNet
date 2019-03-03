using System;
using System.Runtime.InteropServices;
using SiaNet.Backend.MxNetLib.Interop;

// ReSharper disable once CheckNamespace
namespace SiaNet.Backend.MxNetLib
{

    public static class Logging
    {

        #region Fields

        private static readonly string[] OperatorSymbols;

        private static readonly bool ThrowException = false;

        #endregion

        #region Constructors

        static Logging()
        {
            var symbols = new[]
            {
                new { Operator = Operator.Lesser,       Symbol = "<" },
                new { Operator = Operator.Greater,      Symbol = ">" },
                new { Operator = Operator.LesserEqual,  Symbol = "<=" },
                new { Operator = Operator.GreaterEqual, Symbol = ">=" },
                new { Operator = Operator.Equal,        Symbol = "==" },
                new { Operator = Operator.NotEqual,     Symbol = "!=" },
            };

            OperatorSymbols = new string[symbols.Length];
            for (var index = 0; index < symbols.Length; index++)
                OperatorSymbols[index] = symbols[index].Symbol;
        }

        #endregion

        #region Methods

        public static void CHECK<T>(T x, string msg = "")
            where T : class
        {
            if (x != null)
                return;

            var message = string.IsNullOrEmpty(msg) ? $"Check failed: {x} " : $"Check failed: {x} {msg}";
            LOG_FATAL(message);
        }

        public static void CHECK(bool x, string msg = "")
        {
            if (x)
                return;

            var message = string.IsNullOrEmpty(msg) ? $"Check failed: {x} " : $"Check failed: {x} {msg}";
            LOG_FATAL(message);
        }

        private static void CHECK_EQ(string x, string y, string msg = "")
        {
            var error = NativeMethods.MXGetLastError();
            string message;
            if (string.IsNullOrEmpty(msg))
                message = $"Check failed: {x} {OperatorSymbols[(int)Operator.Equal]} {y} {Marshal.PtrToStringAnsi(error) ?? ""}";
            else
                message = $"Check failed: {x} {OperatorSymbols[(int)Operator.Equal]} {y} {msg}";
            //LOG_FATAL(message);
        }

        public static void CHECK_EQ(ulong x, ulong y, string msg = "")
        {
            // dmlc-core/include/dmlc/logging.h
            if (x == y)
                return;

            CHECK_EQ(x.ToString(), y.ToString(), msg);
        }

        public static void CHECK_EQ(uint x, uint y, string msg = "")
        {
            // dmlc-core/include/dmlc/logging.h
            if (x == y)
                return;
            
            CHECK_EQ(x.ToString(), y.ToString(), msg);
        }

        public static void CHECK_EQ(int x, int y, string msg = "")
        {
            // dmlc-core/include/dmlc/logging.h
            if (x == y)
                return;

            CHECK_EQ(x.ToString(), y.ToString(), msg);
        }

        public static void CHECK_EQ(bool x, bool y, string msg = "")
        {
            // dmlc-core/include/dmlc/logging.h
            if (x == y)
                return;

            CHECK_EQ(x.ToString(), y.ToString(), msg);
        }

        public static void CHECK_EQ(Shape x, Shape y, string msg = "")
        {
            // dmlc-core/include/dmlc/logging.h
            if (x == y)
                return;

            CHECK_EQ(x.ToString(), y.ToString(), msg);
        }

        public static void CHECK_NE(int x, int y)
        {
            // dmlc-core/include/dmlc/logging.h
            if (x != y)
                return;

            var error = NativeMethods.MXGetLastError();
            var message = $"Check failed: {x} {OperatorSymbols[(int)Operator.NotEqual]} {y} {Marshal.PtrToStringAnsi(error) ?? ""}";
            LOG_FATAL(message);
        }

        public static void LG(string message)
        {
            Console.WriteLine(message);
        }

        #region Helpers

        private static void LOG_FATAL(string message)
        {
            if (ThrowException)
                throw new MXNetException(message);

            Console.WriteLine(message);
        }

        #endregion

        #endregion

        private enum Operator
        {

            Lesser = 0,

            Greater,

            LesserEqual,

            GreaterEqual,

            Equal,

            NotEqual

        }

    }

}
