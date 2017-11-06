namespace SiaNet.Model.Layers
{
    using System.Dynamic;

    /// <summary>
    /// Global average pooling operation for temporal data.
    /// </summary>
    /// <seealso cref="SiaNet.Model.LayerConfig" />
    public class GlobalAvgPool1D : LayerConfig
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="GlobalAvgPool1D"/> class.
        /// </summary>
        public GlobalAvgPool1D()
        {
            base.Name = "GlobalAvgPool1D";
            base.Params = new ExpandoObject();
        }
    }

    /// <summary>
    /// Global average pooling operation for spatial data.
    /// </summary>
    /// <seealso cref="SiaNet.Model.LayerConfig" />
    public class GlobalAvgPool2D : LayerConfig
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="GlobalAvgPool2D"/> class.
        /// </summary>
        public GlobalAvgPool2D()
        {
            base.Name = "GlobalAvgPool2D";
            base.Params = new ExpandoObject();
        }
    }

    /// <summary>
    /// Global average pooling operation for 3D data (spatial or spatio-temporal).
    /// </summary>
    /// <seealso cref="SiaNet.Model.LayerConfig" />
    public class GlobalAvgPool3D : LayerConfig
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="GlobalAvgPool3D"/> class.
        /// </summary>
        public GlobalAvgPool3D()
        {
            base.Name = "GlobalAvgPool3D";
            base.Params = new ExpandoObject();
        }
    }

}
