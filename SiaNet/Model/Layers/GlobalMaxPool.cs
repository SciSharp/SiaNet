namespace SiaNet.Model.Layers
{
    /// <summary>
    ///     Global max pooling operation for temporal data.
    /// </summary>
    /// <seealso cref="SiaNet.Model.LayerConfig" />
    public class GlobalMaxPool1D : LayerConfig
    {
        /// <summary>
        ///     Initializes a new instance of the <see cref="GlobalMaxPool1D" /> class.
        /// </summary>
        public GlobalMaxPool1D()
        {
            Name = "GlobalMaxPool1D";
        }
    }

    /// <summary>
    ///     Global max pooling operation for spatial data.
    /// </summary>
    /// <seealso cref="SiaNet.Model.LayerConfig" />
    public class GlobalMaxPool2D : LayerConfig
    {
        /// <summary>
        ///     Initializes a new instance of the <see cref="GlobalMaxPool2D" /> class.
        /// </summary>
        public GlobalMaxPool2D()
        {
            Name = "GlobalMaxPool2D";
        }
    }

    /// <summary>
    ///     Global max pooling 3D data (spatial or spatio-temporal).
    /// </summary>
    /// <seealso cref="SiaNet.Model.LayerConfig" />
    public class GlobalMaxPool3D : LayerConfig
    {
        /// <summary>
        ///     Initializes a new instance of the <see cref="GlobalMaxPool3D" /> class.
        /// </summary>
        public GlobalMaxPool3D()
        {
            Name = "GlobalMaxPool3D";
        }
    }
}