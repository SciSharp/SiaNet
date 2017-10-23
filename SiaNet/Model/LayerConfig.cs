using System;
using System.Collections.Generic;
using System.Dynamic;
using System.Linq;
using System.Linq.Expressions;
using System.Text;
using System.Threading.Tasks;

namespace SiaNet.Model
{
    public class LayerConfig
    {
        public string Name { get; set; }

        public dynamic Params { get; set; }
    }
}