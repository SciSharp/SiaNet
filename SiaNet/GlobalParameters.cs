namespace SiaNet
{
    using CNTK;

    /// <summary>
    /// Placeholder for all global parameters. Need to be defined start of the application
    /// </summary>
    public class GlobalParameters
    {
        /// <summary>
        /// Gets or sets the device which will be used. Either CPU or GPU
        /// </summary>
        /// <value>The device.</value>
        public static DeviceDescriptor Device { get; set; }
    }
}
