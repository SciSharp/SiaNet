namespace SiaNet
{
    using SiaNet.Engine;
    using SiaNet.Engine.Layers;
    using System;

    /// <summary>
    /// 
    /// </summary>
    public class Global
    {
        private static IBackend _backend;

        /// <summary>
        /// Get the current backend of the SiaNet. Plaese invoke Global.UseEngine method to set the backend.
        /// </summary>
        /// <value>
        /// The current backend.
        /// </value>
        /// <exception cref="NullReferenceException">Invoke Global.UseEngine() function first to setup the backend and device.</exception>
        public static IBackend CurrentBackend
        {
            get
            {
                if(_backend == null)
                {
                    throw new NullReferenceException("Invoke Global.UseEngine() function first to setup the backend and device.");
                }

                return _backend;
            }
        }

        internal static ActivationFunc ActFunc = null;

        internal static bool UseCudnn { get; set; }

        /// <summary>
        /// Method to set the backend and device type.
        /// <para>
        /// Supported Engine: ArrayFire, TensorSharp, CNTK, TensorFlow, MXNet
        /// </para>
        /// <para>
        /// Supported Device: CPU, CUDA, OpenCL
        /// </para>
        /// <para>
        /// Example use:
        /// <code>
        /// Global.UseEngine(SiaNet.Backend.ArrayFire.SiaNetBackend.Instance, DeviceType.CPU);
        /// </code>
        /// </para>
        /// </summary>
        /// <param name="backend">The backend.</param>
        /// <param name="deviceType">Type of the device.</param>
        /// <param name="cudnn">if set to <c>true</c> [cudnn].</param>
        /// <exception cref="ArgumentException">CuDnn work with CUDA device type</exception>
        public static void UseEngine(IBackend backend, DeviceType deviceType, bool cudnn = false)
        {
            _backend = backend;
            ActFunc = backend.GetActFunc();
            if(cudnn && deviceType != DeviceType.CUDA)
            {
                throw new ArgumentException("CuDnn work with CUDA device type");
            }

            UseCudnn = cudnn;
            _backend.SetDevice(deviceType);
        }
    }
}
