using System;
using System.Collections.Generic;
using System.Globalization;
using System.Runtime.InteropServices;
using System.Text.RegularExpressions;
using SiaNet.Backend.MxNetLib.Interop;
using NDArrayHandle = System.IntPtr;
using StatFunc = System.Func<SiaNet.Backend.MxNetLib.NDArray, SiaNet.Backend.MxNetLib.NDArray>;
using Stat = System.Tuple<int, string, SiaNet.Backend.MxNetLib.NDArray>;

// ReSharper disable once CheckNamespace
namespace SiaNet.Backend.MxNetLib
{

    public class Monitor
    {

        #region Constructors

        public Monitor(int interval)
            : this(interval, ".*")
        {

        }

        public Monitor(int interval, string pattern)
            : this(interval, pattern, DefaultMonitorFunc)
        {

        }

        public Monitor(int interval, string pattern, StatFunc statFunc)
        {
            this.Interval = interval;
            this.Pattern = pattern;
            this.StatFunc = statFunc;

            this.Exes = new List<Executor>();
            this.Stats = new List<Stat>();
        }

        #endregion

        #region Properties

        protected bool Activated
        {
            get;
            private set;
        }

        protected int Interval
        {
            get;
        }

        protected string Pattern
        {
            get;
        }

        protected StatFunc StatFunc
        {
            get;
        }

        protected List<Executor> Exes
        {
            get;
        }

        protected int Step
        {
            get;
            private set;
        }

        protected List<Stat> Stats
        {
            get;
        }

        #endregion

        #region Methods

        public void Install(Executor exe)
        {
            if (exe == null)
                throw new ArgumentNullException(nameof(exe));

            unsafe
            {
                var functionPointer = Marshal.GetFunctionPointerForDelegate(new NativeMethods.ExecutorMonitorCallbackDelegate(executor_callback));
                var gcHandle = GCHandle.Alloc(functionPointer);
                var callbackHandle = GCHandle.Alloc(this);
                var callback = (IntPtr)functionPointer.ToPointer();
                NativeMethods.MXExecutorSetMonitorCallback(exe.Handle, callback, (IntPtr)callbackHandle);
                callbackHandle.Free();
                gcHandle.Free();
            }

            this.Exes.Add(exe);
        }

        public void Tic()
        {
            if (this.Step % this.Interval == 0)
            {
                this.Activated = true;
                this.Stats.Clear();
            }
        }

        public Stat[] Toc()
        {
            var results = new List<Stat>();

            if (this.Activated)
            {
                this.Activated = false;

                foreach (var exe in this.Exes)
                {
                    foreach (var arg in exe.ArgmentArrays)
                        arg.WaitToRead();

                    foreach (var aux in exe.AuxiliaryArrays)
                        aux.WaitToRead();

                    foreach (var pair in exe.ArgmentDictionary())
                    {
                        if (Regex.IsMatch(pair.Key, this.Pattern))
                            this.Stats.Add(new Stat(this.Step, pair.Key, this.StatFunc(pair.Value)));
                    }

                    foreach (var pair in exe.AuxiliaryDictionary())
                    {
                        if (Regex.IsMatch(pair.Key, this.Pattern))
                            this.Stats.Add(new Stat(this.Step, pair.Key, this.StatFunc(pair.Value)));
                    }
                }

                var tmp = results.ToArray();
                results.Clear();
                results.AddRange(this.Stats);
                this.Stats.Clear();
                this.Stats.AddRange(tmp);
            }

            ++this.Step;

            return results.ToArray();
        }

        public void TocPrint()
        {
            var results = this.Toc();
            float[] data = new float[1];
            foreach (var stat in results)
            {
                var ndarray = stat.Item3;

                string str;
                if (ndarray.Size == 1)
                {
                    if (ndarray.GetContext().GetDeviceType() != DeviceType.GPU)
                    {
                        unsafe
                        {
                            var p = (float*)ndarray.GetData();
                            data[0] = p[0];
                        }
                    }
                    else
                    {
                        ndarray.SyncCopyToCPU(data);
                    }

                    str = data[0].ToString(CultureInfo.InvariantCulture);
                }
                else
                {
                    str = ndarray.ToString();
                }

                Logging.LG($"Batch: {stat.Item1} {stat.Item2} {str}");
            }
        }

        #region Helpers

        protected static void executor_callback(string name, NDArrayHandle handle, NDArrayHandle monitorPtr)
        {
            var monitor = GCHandle.FromIntPtr(monitorPtr).Target as Monitor;
            if (monitor != null && monitor.Activated && Regex.IsMatch(name, monitor.Pattern))
            {
                monitor.Stats.Add(new Stat(monitor.Step, name, monitor.StatFunc(new NDArray(handle))));
            }
        }

        private static NDArray DefaultMonitorFunc(NDArray x)
        {
            using (var op = new Operator("norm"))
                return op.PushInput(x).Invoke()[0] / (float)Math.Sqrt(x.Size);
        }

        #endregion

        #endregion

    }

}
