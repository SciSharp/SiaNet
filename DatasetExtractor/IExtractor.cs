using SiaNet.Model;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DatasetExtractor
{
    internal interface IExtractor
    {

        void Download();

        void Extract();
    }
}
