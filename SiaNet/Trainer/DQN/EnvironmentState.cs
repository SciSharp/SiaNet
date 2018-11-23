namespace SiaNet.Trainer.DQN
{
    public class EnvironmentState
    {
        public EnvironmentState(float[] state, float reward, bool isEnded)
        {
            State = state;
            Reward = reward;
            IsEnded = isEnded;
        }

        public bool IsEnded { get; }
        public float Reward { get; }
        public float[] State { get; }
    }
}