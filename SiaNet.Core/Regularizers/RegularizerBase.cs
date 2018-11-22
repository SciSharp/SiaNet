using System;
using CNTK;

namespace SiaNet.Regularizers
{
    /// <summary>
    /// Regularizer allow to apply penalties on layer parameters or layer activity during optimization. These penalties are incorporated in the loss function that the network optimizes
    /// </summary>
    public abstract class RegularizerBase
    {
        private double? _gradientClippingThresholdPerSample;

        protected RegularizerBase(bool gradientClippingWithTruncation)
        {
            GradientClippingWithTruncation = gradientClippingWithTruncation;
        }

        protected RegularizerBase(
            bool gradientClippingWithTruncation,
            double gradientClippingThresholdPerSample) : this(
            gradientClippingWithTruncation)
        {
            GradientClippingThresholdPerSample = gradientClippingThresholdPerSample;
        }

        public bool DoesHaveGradientClippingThresholdPerSample
        {
            get => _gradientClippingThresholdPerSample.HasValue;
        }

        public double GradientClippingThresholdPerSample
        {
            get
            {
                if (_gradientClippingThresholdPerSample.HasValue)
                {
                    return _gradientClippingThresholdPerSample.Value;
                }

                throw new InvalidOperationException();
            }
            protected set => _gradientClippingThresholdPerSample = value;
        }

        public bool GradientClippingWithTruncation { get; protected set; }

        internal abstract AdditionalLearningOptions GetAdditionalLearningOptions();
    }
}