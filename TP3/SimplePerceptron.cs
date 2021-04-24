using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Text;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.Random;
using MathNet.Numerics.Distributions;
using System.Linq;
using MathNet.Numerics;

namespace TP3
{
    public class SimplePerceptron : Perceptron
    {
        public int N { get; private set; }
        public Vector<double> W { get; set; }
        public Func<double, double> ActivationFunction { get; }
        public Func<double, double> ActivationFunctionDerivative { get; }
        public double LearningRate { get; set; }

        public SimplePerceptron(int N, double learningRate, Func<double, double> activationFunction, Func<double, double> activationFunctionDerivative)
        {
            this.N = N;
            this.W = CreateVector.Random<double>(N + 1, new ContinuousUniform(-1d, 1d));
            this.ActivationFunction = activationFunction;
            this.ActivationFunctionDerivative = activationFunctionDerivative;
            this.LearningRate = learningRate;
        }
        private double CalculateError(Vector<double>[] input, double[] desiredOutput, Vector<double> w)
        {
            double sum = 0;
            for(int i = 0; i < input.Length; i++)
            {
                double dif = desiredOutput[i] - ActivationFunction(input[i] * w);
                sum += dif * dif;
            }
            return sum * 0.5d;
        }

        public void Learn(Vector<double>[] trainingInput, double[] desiredOutput, int batch, double minError, int epochs)
        {
            Contract.Requires(trainingInput.Length == desiredOutput.Length);
            Contract.Requires(trainingInput[0].Count == N + 1);

            Vector<double>[] input = new Vector<double>[trainingInput.Length];
            //Agrego el valor 1 al principio del input.
            for (int i = 0; i < input.Length; i++)
            {
                input[i] = Vector<double>.Build.Dense(new double[] { 1 }.Concat(trainingInput[i]).ToArray());
            }
            int p = input.Length;
            Vector<double> w = CreateVector.Random<double>(N + 1, new ContinuousUniform(-1d, 1d));
            Vector<double> deltaW = null;
            double error = 1, error_min = p * 2;
            Vector<double> w_min = W;

            for(int i = 0, n = 0; i < epochs && error_min > minError; i++, n++)
            {
                if (n > 100 * p)
                {
                    w = CreateVector.Random<double>(N + 1, new ContinuousUniform(-1d, 1d));
                    n = 0;
                }
                int[] rand = Combinatorics.GeneratePermutation(input.Length);
                for(int j = 0; j < input.Length; j++)
                {
                    int ix = rand[j];
                    double h = input[ix] * w;
                    double act = ActivationFunction(h);
                    Vector<double> delta = LearningRate * (desiredOutput[ix] - act) * input[ix] * ActivationFunctionDerivative(h);
                    deltaW = deltaW == null ? delta : deltaW + delta;
                    if (j % batch == 0)
                    {
                        w += deltaW;
                        deltaW = null;
                        error = CalculateError(input, desiredOutput, w);
                        if (error < error_min)
                        {
                            error_min = error;
                            w_min = w;
                        }
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
