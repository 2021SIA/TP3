using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using MathNet.Numerics;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.Random;

namespace TP3
{
    public class MultiLayerPerceptron
    {
        public double LearningRate { get; set; }
        private int[] layers;
        private Matrix<double>[] W;
        private Func<double, double>[] g;
        private Func<double, double>[] gprime;

        public MultiLayerPerceptron(int[] layers, double learningRate, Func<double, double>[] g, Func<double, double>[] gprime)
        {
            this.W = new Matrix<double>[layers.Length];
            this.layers = layers;
            this.LearningRate = learningRate;
            this.g = g;
            this.gprime = gprime;
        }
        private double CalculateError(Vector<double>[] input, Vector<double>[] desiredOutput)
        {
            double sum = 0;
            for (int i = 0; i < input.Length; i++)
            {
                double dif = (desiredOutput[i] - Map(input[i])).L2Norm();
                sum += dif * dif;
            }
            return sum * 0.5d;
        }

        public Vector<double> Map(Vector<double> input)
        {
            Vector<double> V = input;
            for (int k = 0; k < layers.Length; k++)
                V = (W[k] * V).Map(g[k]);
            return V;
        }

        public void Learn(Vector<double>[] trainingInput, Vector<double>[] desiredOutput, int batch, double minError, int maxIter = 100)
        {
            Vector<double>[] input = new Vector<double>[trainingInput.Length];
            //Agrego el valor 1 al principio del input.
            for (int i = 0; i < input.Length; i++)
            {
                input[i] = Vector<double>.Build.Dense(new double[]{1}.Concat(trainingInput[i]).ToArray());
            }
            int M = layers.Length;
            //Inicializo los pesos en valores aleatorios.
            for (int i = 0; i < M; i++)
            {
                W[i] = CreateMatrix.Random<double>(layers[i], i == 0 ? input[0].Count : layers[i - 1] + 1);
            }
            Vector<double>[] V = new Vector<double>[M + 1];
            Vector<double>[] delta = new Vector<double>[M];
            Matrix<double>[] deltaW = new Matrix<double>[M];
            Vector<double>[] h = new Vector<double>[M];
            double error = 2 * minError;

            for(int i = 0; i < maxIter && error > minError; i++)
            {
                int[] rand = Combinatorics.GeneratePermutation(input.Length);
                int j;
                for (j = 0; j < input.Length; j++)
                {
                    int index = rand[j];
                    V[0] = input[index];
                    for(int k = 0; k < M; k++)
                    {
                        h[k] = W[k] * V[k];
                        Vector<double> activationOutput = h[k].Map(g[k]);
                        //Agrego el valor 1 al principio de cada salida intermedia.
                        V[k + 1] = k + 1 < M ? Vector<double>.Build.Dense(new double[]{1}.Concat(activationOutput).ToArray()) : h[k].Map(g[k]);
                    }
                    delta[M - 1] = h[M - 1].Map(gprime[M - 1]).PointwiseMultiply(desiredOutput[index] - V[M]);
                    for (int k = M - 1; k > 0; k--)
                        delta[k - 1] = h[k - 1].Map(gprime[k - 1]).PointwiseMultiply(W[k] * delta[k]);
                    for (int k = 0; k < M; k++)
                        deltaW[k] += LearningRate * delta[k].OuterProduct(V[k]);

                    if(j % batch == 0)
                        for (int k = 0; k < M; k++)
                            W[k] += deltaW[k];
                }
                if(j % batch != 0)
                    for (int k = 0; k < M; k++)
                        W[k] += deltaW[k];

                error = CalculateError(input, desiredOutput);
            }
        }
    }
}
