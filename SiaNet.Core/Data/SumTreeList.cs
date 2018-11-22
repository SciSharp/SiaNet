using System;
using System.Collections.Generic;

namespace SiaNet.Data
{
    public class SumTreeList<T>
    {
        protected T[] Data;
        protected int[] Tree;
        protected int Written;

        public SumTreeList(int capacity)
        {
            Capacity = capacity;
            Tree = new int[2 * Capacity - 1];
            Data = new T[Capacity];
        }

        public int Capacity { get; }

        public int Count
        {
            get
            {
                if (Tree != null && Tree.Length > 0)
                {
                    return Tree[0];
                }

                return 0;
            }
        }

        public bool IsFull
        {
            get => Count >= Capacity;
        }

        public void Add(int weight, T data)
        {
            var index = Written + Capacity - 1;
            Data[Written] = data;
            Update(index, weight);

            Written++;

            if (Written >= Capacity)
            {
                Written = 0;
            }
        }

        public void Clear()
        {
            Tree = new int[2 * Capacity - 1];
            Data = new T[Capacity];
            Written = 0;
        }

        public Tuple<int, int, T> Get(int sum)
        {
            var index = Retrieve(0, sum);
            var dataIndex = index - Capacity + 1;

            return new Tuple<int, int, T>(index, Tree[index], Data[dataIndex]);
        }

        public List<Tuple<int, int, T>> ToBatch(int batchSize)
        {
            var list = new List<Tuple<int, int, T>>();
            var segment = Count / (double)batchSize;

            for (var i = 0; i < batchSize; i++)
            {
                var a = segment * i;
                var b = segment * (i + 1);
                var g = Get(RandomGenerator.RandomInt((int)Math.Floor(a), (int)Math.Ceiling(b)));
                list.Add(g);
            }

            return list;
        }

        public void Update(int index, int weight)
        {
            var change = weight - Tree[index];
            Tree[index] = weight;
            Propagate(index, change);
        }

        protected void Propagate(int index, int change)
        {
            var parent = (int) Math.Floor((index - 1) / 2d);
            Tree[parent] += change;

            if (parent > 0)
            {
                Propagate(parent, change);
            }
        }

        protected int Retrieve(int index, int sum)
        {
            var left = 2 * index + 1;
            var right = left + 1;

            if (left >= Tree.Length)
            {
                return index;
            }

            if (sum <= Tree[left])
            {
                return Retrieve(left, sum);
            }

            return Retrieve(right, Tree[left]);
        }
    }
}