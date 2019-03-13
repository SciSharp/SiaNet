namespace SiaNet.Engine.Layers
{
    using System;

    /// <summary>
    /// 
    /// </summary>
    public abstract class ActivationFunc
    {
        IBackend K;

        /// <summary>
        /// Initializes a new instance of the <see cref="ActivationFunc"/> class.
        /// </summary>
        /// <param name="backend">The backend.</param>
        public ActivationFunc(IBackend backend)
        {
            K = backend;
        }

        public virtual Tensor EluForward(float Alpha, Tensor x)
        {
            var keepElements = x > 0;
            var keepElements_Exp = x < 0;
            var d = Alpha * (K.Exp(K.Mul(x, keepElements_Exp)) - 1);
            return K.Mul(x, keepElements) + d;
        }

        public virtual Tensor EluBackward(float Alpha, Tensor x, Tensor outputgrad)
        {
            var keepElements = x > 0;
            var keepElements_Exp = x < 0;
            var d = Alpha * K.Exp(K.Mul(x, keepElements_Exp));
            return outputgrad * d;
        }

        public virtual Tensor HardSigmoidForward(Tensor x)
        {
            throw new NotImplementedException();
        }

        public virtual Tensor HardSigmoidBackward(Tensor x, Tensor outputgrad)
        {
            throw new NotImplementedException();
        }

        public virtual Tensor LeakyReluForward(float Alpha, Tensor x)
        {
            var keepElements = x >= 0;
            return x * keepElements + (Alpha * x * (1 - keepElements));
        }

        public virtual Tensor LeakyReluBackward(float Alpha, Tensor x, Tensor outputgrad)
        {
            var keepElements = x >= 0;
            return outputgrad * (keepElements + (Alpha * (1 - keepElements)));
        }

        public virtual Tensor ReluForward(Tensor x)
        {
            var keepElements = x > 0;
            return x * keepElements + (1 - keepElements) * 0;
        }

        public virtual Tensor ReluBackward(Tensor x, Tensor outputgrad)
        {
            var keepElements = x > 0;
            return outputgrad * (keepElements + (1 - keepElements) * 0);
        }

        public virtual Tensor SoftmaxForward(Tensor x)
        {
            return K.Softmax(x);
        }

        public virtual Tensor SoftmaxBackward(Tensor x, Tensor outputgrad)
        {
            var s = SoftmaxForward(x).Reshape(-1, 1);
            var d = K.Diag(s) - K.Dot(s, s.Transpose());
            return outputgrad * K.Sum(d, -1).Reshape(x.Shape);
        }
    }
}
