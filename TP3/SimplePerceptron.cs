using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Text;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.Random;
using MathNet.Numerics.Distributions;
using System.Linq;

namespace TP3
{
    public class SimplePerceptron
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

        public void Learn(int maxIter, Vector<double>[] trainingInput, double[] desiredOutput)
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
            double error = 1, error_min = p * 2;
            Vector<double> w_min = W;

            for(int i = 0, n = 0; i < maxIter && error > 0; i++, n++)
            {
                if(n > 100 * p)
                {
                    w = CreateVector.Random<double>(N + 1, new ContinuousUniform(-1d, 1d));
                    n = 0;
                }
                int ix = SystemRandomSource.Default.Next(p);
                double h = input[ix] * w;
                double act = ActivationFunction(h);
                w += LearningRate * (desiredOutput[ix] - act) * input[ix] * ActivationFunctionDerivative(h);
                
                error = CalculateError(input, desiredOutput, w);
                if(error < error_min)
                {
                    error_min = error;
                    w_min = w;
                }
            }
            W = w_min;
        }
    }
}
