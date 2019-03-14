namespace SiaNet.Losses
{
    using SiaNet.Engine;

    /// <summary>
    /// Cosine Proximity loss function computes the cosine proximity between predicted value and actual value.
    /// It is same as Cosine Similarity, which is a measure of similarity between two non-zero vectors of an inner product space that measures the cosine of the angle between them. In this case, note that unit vectors are maximally “similar” if they’re parallel and maximally “dissimilar” if they’re orthogonal (perpendicular). 
    /// This is analogous to the cosine, which is unity (maximum value) when the segments subtend a zero angle and zero (uncorrelated) when the segments are perpendicular.
    /// </summary>
    /// <seealso cref="SiaNet.Losses.BaseLoss" />
    public class CosineProximity : BaseLoss
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="CosineProximity"/> class.
        /// </summary>
        public CosineProximity()
            : base("cosine_proximity")
        {

        }

        /// <summary>
        /// Forwards the inputs and calculate the loss.
        /// </summary>
        /// <param name="preds">The predicted result.</param>
        /// <param name="labels">The true result.</param>
        /// <returns></returns>
        public override Tensor Forward(Tensor preds, Tensor labels)
        {
            return -1 * _cossine_sim(preds, labels);
        }

        /// <summary>
        /// Backpropagation method to calculate gradient of the loss function
        /// </summary>
        /// <param name="preds">The predicted result.</param>
        /// <param name="labels">The true result.</param>
        /// <returns></returns>
        public override Tensor Backward(Tensor preds, Tensor labels)
        {
            var y_true = K.Max(K.Sum(labels, -1), -1) / (K.Abs(preds * K.Abs(labels)));
            var y_pred = K.Max(K.Sum(preds, -1), -1) / K.Square(K.Abs(preds));

            return y_true + _cossine_sim(preds, labels) * y_pred;
        }

        private Tensor _cossine_sim(Tensor preds, Tensor labels)
        {
            var y_true = K.L2Normalize(labels, -1);
            var y_pred = K.L2Normalize(preds, -1);
            return K.Sum(y_true * y_pred, -1);
        }
    }
}
