using MathNet.Numerics;
using MathNet.Numerics.LinearAlgebra;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace TP3
{
    public class PerceptronUtils
    {
        public struct ClassifierMetrics
        {
            public double Accuracy { get; set; }
            public double Precision { get; set; }
            public double Recall { get; set; }
            public double F1Score { get; set; }
        }

        public static ClassifierMetrics GetClassifierMetrics(Perceptron p, IEnumerable<Vector<double>> input, IEnumerable<Vector<double>> output)
        {
            Matrix<double> confusionMatrix = Matrix<double>.Build.Dense(2, 2);
            for (int i = 0; i < input.Count(); i++)
            {
                Vector<double> result = p.Map(input.ElementAt(i));
                int row = output.ElementAt(i)[0] == 1 ? 0 : 1;
                int col = result[0] == 1 ? 0 : 1;
                confusionMatrix[row, col] += 1;
            }
            double accuracy = (confusionMatrix[0, 0] + confusionMatrix[1, 1]) / confusionMatrix.ReduceColumns((val, c) => val + c).Aggregate(0.0, (val, c) => val + c);
            double precision = confusionMatrix[0, 0] != 0 ? confusionMatrix[0, 0] / (confusionMatrix[0, 0] + confusionMatrix[1, 0]) : 0;
            double recall = confusionMatrix[0, 0] != 0 ? confusionMatrix[0, 0] / (confusionMatrix[0, 0] + confusionMatrix[0, 1]) : 0;
            double f1Score = precision != 0 && recall != 0 ? (2 * precision * recall) / (precision + recall) : 0;
            return new ClassifierMetrics() { Accuracy = accuracy,Precision = precision,Recall = recall, F1Score = f1Score };
        }
        public static double Sigmoid(double value)
        {
            return 1.0 / (1.0 + Math.Exp(-value));
        }
        public static List<Vector<double>> Normalize(IEnumerable<Vector<double>> values, double rangeBottom, double rangeTop)
        {
            if (values.Count() == 0) return values.ToList();
            var max = values.Max(v => v.Max());
            var min = values.Min(v => v.Min());
            return values.Select(v => Normalize(v, max, min, rangeBottom, rangeTop)).ToList();
        }
        public static Vector<double> Normalize(Vector<double> vector, double max, double min, double rangeBottom, double rangeTop)
        {
            return vector.Map(value => ((value - min) / (max - min)) * (rangeTop - rangeBottom) + rangeBottom);
        }
        public static (List<(Vector<double>[], Vector<double>[])>, List<(Vector<double>[], Vector<double>[])>) GetNTestGroups(IEnumerable<Vector<double>> input, IEnumerable<Vector<double>> output, int testSize, int N)
        {
            var inputGroups = new List<(Vector<double>[], Vector<double>[])>();
            var outputGroups = new List<(Vector<double>[], Vector<double>[])>();
            if (testSize == 0)
            {
                inputGroups.Add((input.ToArray(), new Vector<double>[0]));
                outputGroups.Add((output.ToArray(), new Vector<double>[0]));
            }
            else
            {
                for (int i = 0; i < N; i++)
                {
                    int[] rand = Combinatorics.GeneratePermutation(input.Count());
                    var testInput = rand.Take(testSize).Select(input.ElementAt);
                    var trainInput = rand.Skip(testSize).Select(input.ElementAt);
                    var testOutput = rand.Take(testSize).Select(output.ElementAt);
                    var trainOutput = rand.Skip(testSize).Select(output.ElementAt);
                    inputGroups.Add((testInput.ToArray(), trainInput.ToArray()));
                    outputGroups.Add((testOutput.ToArray(), trainOutput.ToArray()));
                }
            }
            return (inputGroups,outputGroups);
        }
    }
}
