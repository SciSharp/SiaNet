namespace SiaNet.Model.Layers
{
    /// <summary>
    ///     Global average pooling operation for temporal data.
    /// </summary>
    /// <seealso cref="SiaNet.Model.LayerConfig" />
    public class GlobalAvgPool1D : LayerConfig
    {
        /// <summary>
        ///     Initializes a new instance of the <see cref="GlobalAvgPool1D" /> class.
        /// </summary>
        public GlobalAvgPool1D()
        {
            Name = "GlobalAvgPool1D";
        }
    }

    /// <summary>
    ///     Global average pooling operation for spatial data.
    /// </summary>
    /// <seealso cref="SiaNet.Model.LayerConfig" />
    public class GlobalAvgPool2D : LayerConfig
    {
        /// <summary>
        ///     Initializes a new instance of the <see cref="GlobalAvgPool2D" /> class.
        /// </summary>
        public GlobalAvgPool2D()
        {
            Name = "GlobalAvgPool2D";
        }
    }

    /// <summary>
    ///     Global average pooling operation for 3D data (spatial or spatio-temporal).
    /// </summary>
    /// <seealso cref="SiaNet.Model.LayerConfig" />
    public class GlobalAvgPool3D : LayerConfig
    {
        /// <summary>
        ///     Initializes a new instance of the <see cref="GlobalAvgPool3D" /> class.
        /// </summary>
        public GlobalAvgPool3D()
        {
            Name = "GlobalAvgPool3D";
        }
    }
}