namespace SiaNet.Model.Layers
{
    using System.Dynamic;

    /// <summary>
    /// Global max pooling operation for temporal data.
    /// </summary>
    /// <seealso cref="SiaNet.Model.LayerConfig" />
    public class GlobalMaxPool1D : LayerConfig
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="GlobalMaxPool1D"/> class.
        /// </summary>
        public GlobalMaxPool1D()
        {
            base.Name = "GlobalMaxPool1D";
            base.Params = new ExpandoObject();
        }
    }

    /// <summary>
    /// Global max pooling operation for spatial data.
    /// </summary>
    /// <seealso cref="SiaNet.Model.LayerConfig" />
    public class GlobalMaxPool2D : LayerConfig
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="GlobalMaxPool2D"/> class.
        /// </summary>
        public GlobalMaxPool2D()
        {
            base.Name = "GlobalMaxPool2D";
            base.Params = new ExpandoObject();
        }
    }

    /// <summary>
    /// Global max pooling 3D data (spatial or spatio-temporal).
    /// </summary>
    /// <seealso cref="SiaNet.Model.LayerConfig" />
    public class GlobalMaxPool3D : LayerConfig
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="GlobalMaxPool3D"/> class.
        /// </summary>
        public GlobalMaxPool3D()
        {
            base.Name = "GlobalMaxPool3D";
            base.Params = new ExpandoObject();
        }
    }

}
