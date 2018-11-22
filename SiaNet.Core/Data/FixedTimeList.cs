using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;

namespace SiaNet.Data
{
    public class FixedTimeList<T> : IEnumerable<T>
    {
        protected List<Tuple<T, DateTime>> UnderlyingList = new List<Tuple<T, DateTime>>();

        public FixedTimeList(TimeSpan timeLimit)
        {
            TimeLimit = timeLimit;
        }

        public int Count
        {
            get => UnderlyingList.Count;
        }
        
        public TimeSpan TimeLimit { get; }

        /// <inheritdoc />
        IEnumerator IEnumerable.GetEnumerator()
        {
            return GetEnumerator();
        }

        /// <inheritdoc />
        public IEnumerator<T> GetEnumerator()
        {
            return UnderlyingList.OrderByDescending(t => t.Item2).Select(t => t.Item1).GetEnumerator();
        }

        public void Add(T item, DateTime time)
        {
            UnderlyingList.Insert(0, new Tuple<T, DateTime>(item, time));

            var maxTime = UnderlyingList.Max(t => t.Item2);
            foreach (var t in UnderlyingList.Where(t => maxTime - t.Item2 >= TimeLimit).ToArray())
            {
                UnderlyingList.Remove(t);
            }
        }

        public void Clear()
        {
            UnderlyingList.Clear();
        }

        public bool Contains(T item)
        {
            return UnderlyingList.Any(t => t.Item1.Equals(item));
        }

        public T[] ToArray()
        {
            return UnderlyingList.OrderByDescending(t => t.Item2).Select(t => t.Item1).ToArray();
        }

        public Dictionary<DateTime, T[]> ToDictionary()
        {
            return UnderlyingList.OrderByDescending(t => t.Item2).GroupBy(t => t.Item2)
                .ToDictionary(group => group.Key, group => group.Select(t => t.Item1).ToArray());
        }

        public List<T> ToList()
        {
            return UnderlyingList.OrderByDescending(t => t.Item2).Select(t => t.Item1).ToList();
        }
    }
}