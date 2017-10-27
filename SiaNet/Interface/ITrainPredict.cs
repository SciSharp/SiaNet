using SiaNet.Model;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SiaNet.Interface
{
    public interface ITrainPredict
    {
        Dictionary<string, List<double>> Train(object trainData, object validationData, int epoches, int batchSize, On_Epoch_Start OnEpochStart, On_Epoch_End OnEpochEnd);

        IList<float> Evaluate(object testData);
    }
}
