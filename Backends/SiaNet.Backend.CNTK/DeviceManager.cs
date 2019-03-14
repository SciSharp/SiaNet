using CNTK;
using System;
using System.Collections.Generic;
using System.Text;

namespace SiaNet.Backend.CNTKLib
{
    public class DeviceManager
    {
        public static DeviceDescriptor Current { get; set; } = DeviceDescriptor.CPUDevice;
    }
}
