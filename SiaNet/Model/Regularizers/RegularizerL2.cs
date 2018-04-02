using CNTK;

namespace SiaNet.Model.Regularizers
{
    /// <inheritdoc />
    public class RegularizerL2 : RegularizerBase
    {
        /// <inheritdoc />
        public RegularizerL2(double l2 = 0.01) : base(false)
        {
            L2 = l2;
        }

        /// <inheritdoc />
        public RegularizerL2(double l2, bool gradientClippingWithTruncation) : base(gradientClippingWithTruncation)
        {
            L2 = l2;
        }

        /// <inheritdoc />
        public RegularizerL2(double l2, bool gradientClippingWithTruncation, double gradientClippingThresholdPerSample)
            : base(gradientClippingWithTruncation, gradientClippingThresholdPerSample)
        {
            L2 = l2;
        }

        public double L2 { get; }


        /// <inheritdoc />
        internal override AdditionalLearningOptions GetAdditionalLearningOptions()
        {
            var options = new AdditionalLearningOptions
            {
                l2RegularizationWeight = L2,
                gradientClippingWithTruncation = GradientClippingWithTruncation
            };

            if (DoesHaveGradientClippingThresholdPerSample)
            {
                options.gradientClippingThresholdPerSample = GradientClippingThresholdPerSample;
            }

            return options;
        }
    }
}