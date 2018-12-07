using SiaNet.Backend;
using System;
using System.Collections.Generic;
using System.Text;

namespace SiaNet
{
    public class GlobalParam
    {
        public static Context Device { get; set; }

        public static bool UseCudnn { get; set; }
    }
}
