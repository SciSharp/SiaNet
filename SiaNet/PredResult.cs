using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SiaNet
{
    /// <summary>
    /// Class to hold the prediction result
    /// </summary>
    public class PredResult
    {
        /// <summary>
        /// Gets or sets the name of the result like detected object name, image class name.
        /// </summary>
        /// <value>
        /// The name.
        /// </value>
        public string Name { get; set; }

        /// <summary>
        /// Gets or sets the score of the accuracy for the detected or classified object
        /// </summary>
        /// <value>
        /// The score.
        /// </value>
        public double Score { get; set; }

        /// <summary>
        /// Gets or sets the detection window of the detected object in an image
        /// </summary>
        /// <value>
        /// The b box.
        /// </value>
        public Rectangle BBox { get; set; }

    }
    
}
