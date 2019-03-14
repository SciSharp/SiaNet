namespace SiaNet.Initializers
{
    /// <summary>
    /// Initializer that generates tensors initialized to 1.
    /// </summary>
    /// <seealso cref="SiaNet.Initializers.Constant" />
    public class Ones : Constant
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="Ones"/> class.
        /// </summary>
        public Ones()
            : base("ones", 1)
        {

        }
    }
}
