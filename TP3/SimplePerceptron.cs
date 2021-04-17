﻿using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Text;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.Random;
using MathNet.Numerics.Distributions;

namespace TP3
{
    public class SimplePerceptron
    {
        public int N { get; private set; }
        public Vector<double> W { get; set; }
        public Func<double, double> ActivationFunction { get; }
        public Func<double, double> ActivationFunctionDerivative { get; }
        public double LearningRate { get; set; }

        private double CalculateError(Vector<double>[] input, double[] desiredOutput, Vector<double> w)
        {
            double sum = 0;
            for(int i = 0; i < input.Length; i++)
            {
                double dif = desiredOutput[i] - input[i] * w;
                sum += dif * dif;
            }
            return sum * 0.5d;
        }

        public void Learn(int maxIter, Vector<double>[] input, double[] desiredOutput)
        {
            Contract.Requires(input.Length == desiredOutput.Length);
            Contract.Requires(input[0].Count == N + 1);

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
                double act = ActivationFunction(input[ix] * w);
                w += LearningRate * (desiredOutput[ix] - act) * input[ix] * ActivationFunctionDerivative(act);
                
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
