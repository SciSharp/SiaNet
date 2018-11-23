using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;

namespace SiaNet.Data
{
    public class FixedSizeList<T> : IList<T>
    {
        protected readonly List<T> UnderlyingList = new List<T>();
        public bool Reverse { get; }
        public FixedSizeList(int capacity, bool reverseOrder = true)
        {
            Capacity = capacity;
            Reverse = reverseOrder;
        }

        public int Capacity { get; }

        public bool IsFull
        {
            get => Count >= Capacity;
        }

        /// <inheritdoc />
        public void Add(T item)
        {
            if (Reverse)
            {
                Insert(0, item);
            }
            else
            {
                Insert(IsFull ? Capacity - 1 : Count, item);
            }
        }


        public void Clear()
        {
            UnderlyingList.Clear();
        }

        /// <inheritdoc />
        public bool Contains(T item)
        {
            return UnderlyingList.Contains(item);
        }

        /// <inheritdoc />
        public void CopyTo(T[] array, int arrayIndex)
        {
            UnderlyingList.CopyTo(array, arrayIndex);
        }

        public int Count
        {
            get => UnderlyingList.Count;
        }

        /// <inheritdoc />
        public bool IsReadOnly
        {
            get => false;
        }

        /// <inheritdoc />
        public bool Remove(T item)
        {
            return UnderlyingList.Remove(item);
        }

        /// <inheritdoc />
        IEnumerator IEnumerable.GetEnumerator()
        {
            return GetEnumerator();
        }

        /// <inheritdoc />
        public IEnumerator<T> GetEnumerator()
        {
            return UnderlyingList.GetEnumerator();
        }

        /// <inheritdoc />
        public int IndexOf(T item)
        {
            return UnderlyingList.IndexOf(item);
        }

        /// <inheritdoc />
        public void Insert(int index, T item)
        {
            if (UnderlyingList.Count == Capacity)
            {
                if (Reverse)
                {
                    UnderlyingList.RemoveAt(UnderlyingList.Count - 1);
                }
                else
                {
                    UnderlyingList.RemoveAt(0);
                }
            }

            UnderlyingList.Insert(index, item);
        }

        /// <inheritdoc />
        public T this[int index]
        {
            get => UnderlyingList[index];
            set => UnderlyingList[index] = value;
        }

        /// <inheritdoc />
        public void RemoveAt(int index)
        {
            UnderlyingList.RemoveAt(index);
        }

        public T[] ToArray()
        {
            return UnderlyingList.ToArray();
        }

        public List<T> ToList()
        {
            return UnderlyingList.ToList();
        }

        public T[] ToRandomBatch(int batchSize)
        {
            var len = Math.Min(Count, batchSize);
            int start = RandomGenerator.RandomIntInclusive(0, Count - batchSize);
            return UnderlyingList.Skip(start).Take(len).ToArray();
        }

        public T[] ToShuffledBatch(int batchSize)
        {
            return ToShuffledArray().Take(batchSize).ToArray();
        }

        public T[] ToShuffledArray()
        {
            var array = ToArray();

            for (var i = array.Length - 1; i >= 0; i--)
            {
                var r = RandomGenerator.RandomInt(0, i);
                var features = array[i];
                array[i] = array[r];
                array[r] = features;
            }

            return array;
        }
    }
}