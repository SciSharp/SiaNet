using System;
using System.Collections.Generic;
using System.Text;
using System.Threading;

namespace SiaNet
{
    public static class UUID
    {
        private static Dictionary<string, int> CurrentIndexes = new Dictionary<string, int>();
        private static int counter = 0;

        public static void Reset()
        {
            CurrentIndexes = new Dictionary<string, int>();
        }

        private static int Next(string name)
        {
            if (!CurrentIndexes.ContainsKey(name))
            {
                CurrentIndexes.Add(name, 0);
            }

            CurrentIndexes[name] = counter++;

            return CurrentIndexes[name];
        }

        public static string GetID(string name)
        {
            string result = "";
            result = string.Format("{0}_{1}", name.ToLower(), Next(name));

            return result;
        }
    }
}
