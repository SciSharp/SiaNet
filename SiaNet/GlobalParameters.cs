namespace SiaNet
{
    using CNTK;
    using SiaNet.Common;
    using System.Linq;

    /// <summary>
    /// Placeholder for all global parameters. Need to be defined start of the application
    /// </summary>
    public class GlobalParameters
    {
        private static DeviceDescriptor device;

        /// <summary>
        /// Gets or sets the device which will be used. Either CPU or GPU
        /// </summary>
        /// <value>The device.</value>
        public static DeviceDescriptor Device
        {
            get
            {
                if (device == null)
                {
                    var gpuList = DeviceDescriptor.AllDevices().Where(x => (x.Type == DeviceKind.GPU)).ToList();
                    if (gpuList.Count > 0)
                        device = gpuList[0];
                    else
                        device = DeviceDescriptor.CPUDevice;
                }

                return device;
            }
            set
            {
                device = value;
                if (device.Type == DeviceKind.CPU)
                {
                    Logging.WriteTrace("Selected device: CPU. Please use GPU for better performance.");
                }
                else
                {
                    Logging.WriteTrace("Selected device: GPU.");
                }
            }
        }
    }
}
