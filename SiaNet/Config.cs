using System;
using System.Collections.Generic;
using System.IO;
using System.Text;

namespace SiaNet
{
    public class Config
    {
        public string Environment { get; set; }

        public bool UseGpu { get; set; }

        public bool UseCudnn { get; set; }

        public string CudaPath { get; set; }

        public static Config GetConfig()
        {
            string json = File.ReadAllText("config.json");
            Config config = Newtonsoft.Json.JsonConvert.DeserializeObject<Config>(json);
            return config;
        }
    }
}
