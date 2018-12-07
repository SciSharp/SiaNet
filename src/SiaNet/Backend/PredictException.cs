using System;

// ReSharper disable once CheckNamespace
namespace SiaNet.Backend
{

    /// <summary>
    /// The exception that is thrown when occurs error of prediction. This class cannot be inherited.
    /// </summary>
    public sealed class PredictException : Exception
    {

        #region Constructors

        /// <summary>
        /// Initializes a new instance of the <see cref="PredictException"/> class.
        /// </summary>
        public PredictException()
            : base()
        {
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="PredictException"/> class with a specified error message.
        /// </summary>
        /// <param name="message">The error message that explains the reason for the exception.</param>
        public PredictException(string message)
            : base(message)
        {
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="PredictException"/> class with a specified error message and a reference to the inner exception that is the cause of this exception.
        /// </summary>
        /// <param name="message">The error message that explains the reason for the exception.</param>
        /// <param name="innerException">The name of the parameter that caused the current exception.</param>
        public PredictException(string message, Exception innerException)
            : base(message, innerException)
        {
        }

        #endregion

    }

}
