using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SiaNet.Events
{
    /// <summary>
    /// Event delegation on start of training
    /// </summary>
    public delegate void On_Training_Start();

    /// <summary>
    /// Event delegation at end of training
    /// </summary>
    /// <param name="trainingResult">The training result.</param>
    public delegate void On_Training_End(Dictionary<string, List<double>> trainingResult);

    /// <summary>
    /// Event delegation on start of epoch
    /// </summary>
    /// <param name="epoch">The epoch.</param>
    public delegate void On_Epoch_Start(int epoch);

    /// <summary>
    /// Event delegation at end of epoch
    /// </summary>
    /// <param name="epoch">The epoch.</param>
    /// <param name="samplesSeen">The samples seen.</param>
    /// <param name="loss">The loss.</param>
    /// <param name="metrics">The metrics.</param>
    public delegate void On_Epoch_End(int epoch, uint samplesSeen, double loss, Dictionary<string, double> metrics);

    /// <summary>
    /// Event delegation at start of batch
    /// </summary>
    /// <param name="epoch">The epoch.</param>
    /// <param name="batchNumber">The batch number.</param>
    public delegate void On_Batch_Start(int epoch, int batchNumber);

    /// <summary>
    /// Event delegation at end of batch
    /// </summary>
    /// <param name="epoch">The epoch.</param>
    /// <param name="batchNumber">The batch number.</param>
    /// <param name="samplesSeen">The samples seen.</param>
    /// <param name="loss">The loss.</param>
    /// <param name="metrics">The metrics.</param>
    public delegate void On_Batch_End(int epoch, int batchNumber, uint samplesSeen, double loss, Dictionary<string, double> metrics);
}
