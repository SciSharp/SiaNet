namespace SiaNet.Layers.Activations
{
    using SiaNet.Engine;

    public class HardSigmoid : BaseLayer
    {
        public HardSigmoid()
            : base("hard_sigmoid")
        {
        }

        public override void Forward(Tensor x)
        {
            base.Forward(x);
            Output = Global.ActFunc.HardSigmoidForward(x);
        }

        public override void Backward(Tensor outputgrad)
        {
            Input.Grad = Global.ActFunc.HardSigmoidBackward(Input.Data, outputgrad);
        }
    }
}
