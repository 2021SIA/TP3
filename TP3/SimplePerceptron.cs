using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Text;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.Random;
using MathNet.Numerics.Distributions;
using System.Linq;
using MathNet.Numerics;
using Accord.Math.Optimization;
using System.IO;

namespace TP3
{
    public class SimplePerceptron : Perceptron
    {
        public int N { get; private set; }
        public Vector<double> W { get; set; }
        public Func<double, double> ActivationFunction { get; }
        public Func<double, double> ActivationFunctionDerivative { get; }
        public double LearningRate { get; set; }
        public bool AdaptiveLearningRate { get; set; }

        public SimplePerceptron(int N, double learningRate, Func<double, double> activationFunction, Func<double, double> activationFunctionDerivative, bool adaptiveLearningRate)
        {
            this.N = N;
            this.W = CreateVector.Random<double>(N + 1, new ContinuousUniform(-1d, 1d));
            this.ActivationFunction = activationFunction;
            this.ActivationFunctionDerivative = activationFunctionDerivative;
            this.LearningRate = learningRate;
            this.AdaptiveLearningRate = adaptiveLearningRate;
        }
        private double CalculateError(Vector<double>[] input, double[] desiredOutput, Vector<double> w)
        {
            if (input.Length == 0) return 0;

            if(input[0].Count < w.Count)
            {
                //Agrego el valor 1 al principio del input.
                for (int i = 0; i < input.Length; i++)
                    input[i] = Vector<double>.Build.DenseOfEnumerable(new double[] { 1 }.Concat(input[i]));
            }
            double sum = 0;
            for (int i = 0; i < input.Length; i++)
            {
                double dif = desiredOutput[i] - ActivationFunction(input[i] * w);
                sum += dif * dif;
            }
            return sum * 0.5d;
        }
        public double CalculateError(Vector<double>[] input, Vector<double>[] desiredOutput) => CalculateError(input, desiredOutput.Select(o => o.At(0)).ToArray(), W);


        private double optimizing(
            Vector<double> w,
            Vector<double>[] input,
            double[] desiredTrainingOutput,
            int batch,
            int[] rand)
        {
            Func<double, double> function = x => loop(w, input, desiredTrainingOutput, batch, x, rand);
            BrentSearch search = new BrentSearch(function, 0, 1);
            bool success = search.Minimize();
            double min = search.Solution;

            return min;
        }

        private double loop(
            Vector<double> w,
            Vector<double>[] input,
            double[] desiredTrainingOutput,
            int batch,
            double lr,
            int[] rand)
        {

            Vector<double> deltaW = null;
            double error = 0, error_min = -1;
            for (int j = 0; j < input.Length; j++)
            {
                int ix = rand[j];
                double h = input[ix] * w;
                double act = ActivationFunction(h);
                Vector<double> delta = lr * (desiredTrainingOutput[ix] - act) * input[ix] * ActivationFunctionDerivative(h);
                deltaW = deltaW == null ? delta : deltaW + delta;
                if (j % batch == 0)
                {
                    w += deltaW;
                    deltaW = null;
                    error = CalculateError(input, desiredTrainingOutput, w);
                    if (error_min == -1 || error < error_min)
                        error_min = error;
                }
            }
            return error_min;
        }




        public void Learn(
            Vector<double>[] trainingInput,
            Vector<double>[] trainingOutput,
            Vector<double>[] testInput,
            Vector<double>[] testOutput,
            int batch,
            double minError,
            int epochs)
        {
            Contract.Requires(trainingInput.Length == trainingOutput.Length);
            Contract.Requires(testInput.Length == testOutput.Length);
            Contract.Requires(trainingInput[0].Count == N + 1);
            Contract.Requires(testInput[0].Count == N + 1);

            double[] desiredTrainingOutput = trainingOutput.Select(o => o.At(0)).ToArray();
            double[] desiredTestOutput = trainingOutput.Select(o => o.At(0)).ToArray();

            //Agrego el valor 1 al principio del input.
            Vector<double>[] input = new Vector<double>[trainingInput.Length];
            for (int i = 0; i < input.Length; i++)
            {
                input[i] = Vector<double>.Build.DenseOfEnumerable(new double[] { 1 }.Concat(trainingInput[i]));
            }
            int p = input.Length;
            Vector<double> w = CreateVector.Random<double>(N + 1, new ContinuousUniform(-1d, 1d));
            Vector<double> deltaW = null;
            double error = 1, error_min = p * 2;
            Vector<double> w_min = w;

            for(int i = 0, n = 0; i < epochs && error_min > minError; i++, n++)
            {
                if (n > 100 * p)
                {
                    w = CreateVector.Random<double>(N + 1, new ContinuousUniform(-1d, 1d));
                    n = 0;
                }
                
                int[] rand = Combinatorics.GeneratePermutation(input.Length);
                double lr = this.AdaptiveLearningRate ? optimizing( w, input, desiredTrainingOutput, batch, rand) : LearningRate;
                int j;
                for (j = 0; j < input.Length; j++)
                {
                    int ix = rand[j];
                    double h = input[ix] * w;
                    double act = ActivationFunction(h);
                    Vector<double> delta = lr * (desiredTrainingOutput[ix] - act) * input[ix] * ActivationFunctionDerivative(h);
                    deltaW = deltaW == null ? delta : deltaW + delta;
                    if (j % batch == 0)
                    {
                        w += deltaW;
                        deltaW = null;
                        error = CalculateError(input, desiredTrainingOutput, w);
                        if (error < error_min)
                        {
                            error_min = error;
                            w_min = w;
                        }
                    }
                }
                if (j % batch != 0)
                {
                    w += deltaW;
                    deltaW = null;
                    error = CalculateError(input, desiredTrainingOutput, w);
                    if (error < error_min)
                    {
                        error_min = error;
                        w_min = w;
                    }
                }
            }
            W = w_min;
        }

        public Vector<double> Map(Vector<double> input)
        {
            Vector<double> aux = Vector<double>.Build.Dense(new double[] { 1 }.Concat(input).ToArray());
            //Agrego el valor 1 al principio del input.
            return Vector<double>.Build.DenseOfArray(new double[] { ActivationFunction(aux * W) });
        }
    }
}
