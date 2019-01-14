using System;
using System.Collections.Generic;
using System.Text;

namespace SiaNet
{
    public static class UUID
    {
        private static Dictionary<string, int> CurrentIndexes = new Dictionary<string, int>();
        private static int counter = 0;
        private static object lockObject = new object();

        public static void Reset()
        {
            CurrentIndexes = new Dictionary<string, int>();
        }

        private static int Next(string name)
        {
            lock (lockObject)
            {
                if (!CurrentIndexes.ContainsKey(name))
                {
                    CurrentIndexes.Add(name, 0);
                }

                CurrentIndexes[name] += 1;
            }

            return CurrentIndexes[name];
        }

        public static string GetID(string name)
        {
            string result = "";
            lock (lockObject)
            {
                result = string.Format("{0}_{1}", name.ToLower(), counter++);
            }

            return result;
        }
    }
}
