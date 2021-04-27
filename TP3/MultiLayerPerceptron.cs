using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using MathNet.Numerics;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.Random;
using Accord.Math.Optimization;


namespace TP3
{
    public class MultiLayerPerceptron : Perceptron
    {
        public double LearningRate { get; set; }
        public bool AdaptiveLearningRate { get; set; }
        private int[] layers;
        private Matrix<double>[] W;
        private Func<double, double>[] g;
        private Func<double, double>[] gprime;


        public MultiLayerPerceptron(int[] layers, double learningRate, Func<double, double>[] g, Func<double, double>[] gprime, bool adaptiveLearningRate)
        {
            this.W = new Matrix<double>[layers.Length];
            this.layers = layers;
            this.LearningRate = learningRate;
            this.AdaptiveLearningRate = adaptiveLearningRate;
            this.g = g;
            this.gprime = gprime;
        }
        public double CalculateError(Vector<double>[] input, Vector<double>[] desiredOutput) => CalculateError(input, desiredOutput, W);
        private double CalculateError(Vector<double>[] input, Vector<double>[] desiredOutput, Matrix<double>[] w)
        {
            double sum = 0;
            for (int i = 0; i < input.Length; i++)
            {
                double dif = (desiredOutput[i] - Map(input[i],w)).L2Norm();
                sum += dif * dif;
            }
            return sum * 0.5d;
        }

        public Vector<double> Map(Vector<double> input) => Map(input, W);
        private Vector<double> Map(Vector<double> input, Matrix<double>[] w)
        {
            Vector<double> V = input;
            for (int k = 0; k < layers.Length; k++)
            {
                //Agrego el valor 1 al principio de cada salida intermedia.
                if (k > 0 || (w[k].ColumnCount != V.Count))
                    V = Vector<double>.Build.DenseOfEnumerable(new double[] { 1 }.Concat(V));
                V = (w[k] * V).Map(g[k]);
            }
            return V;
        }
        private double optimizing(Vector<double>[] input,
            Vector<double>[] V,
            Matrix<double>[] w,
            int M,
            Vector<double>[] h,
            Vector<double>[] delta,
            Vector<double>[] trainingOutput,
            Matrix<double>[] deltaW,
            int batch,
            double error,
            int[] rand)
        {
            Func<double, double> function = x => loop(input, V, M, h, delta, trainingOutput, deltaW, batch, error, x, rand, w);
            BrentSearch search = new BrentSearch(function, 0, 1);
            bool success = search.Minimize();   
            double min = search.Solution;

            return min;
        }


        

        private double loop(
            Vector<double>[] input,
            Vector<double>[] V,
            int M,
            Vector<double>[] h,
            Vector<double>[] delta,
            Vector<double>[] trainingOutput,
            Matrix<double>[] deltaW,
            int batch,
            double error,
            double lr,
            int[] rand,
            Matrix<double>[] w)
        {
            int j;
            double error_min = -1;
            for (j = 0; j < input.Length; j++)
            {
                int index = rand[j];
                V[0] = input[index];
                for(int k = 0; k < M; k++)
                {
                    h[k] = w[k] * V[k];
                    Vector<double> activationOutput = h[k].Map(g[k]);
                    //Agrego el valor 1 al principio de cada salida intermedia.
                    V[k + 1] = k + 1 < M ? Vector<double>.Build.DenseOfEnumerable(new double[]{1}.Concat(activationOutput)) : activationOutput;
                }
                delta[M - 1] = h[M - 1].Map(gprime[M - 1]).PointwiseMultiply(trainingOutput[index] - V[M]);
                for (int k = M - 1; k > 0; k--)
                {
                    var aux = w[k].TransposeThisAndMultiply(delta[k]);
                    //Salteo el primer valor ya que corresponde al delta de la neurona extra para el umbral (siempre tiene que tener activacion igual a 1).
                    aux = Vector<double>.Build.DenseOfEnumerable(aux.Skip(1));
                    delta[k - 1] = h[k - 1].Map(gprime[k - 1]).PointwiseMultiply(aux);

                }
                
                for (int k = 0; k < M; k++)
                {
                    if(deltaW[k] != null)
                        deltaW[k] += lr * delta[k].OuterProduct(V[k]);
                    else
                        deltaW[k] = lr * delta[k].OuterProduct(V[k]);
                }

                if(j % batch == 0)
                {
                    for (int k = 0; k < M; k++)
                        w[k] += deltaW[k];
                    error = CalculateError(input, trainingOutput, w);
                    if (error_min == -1 || error < error_min)
                        error_min = error;
                    //Reinicio los deltaW para el proximo lote.
                    Array.Fill(deltaW, null);
                }
            }
            if (j % batch != 0)
            {
                for (int k = 0; k < M; k++)
                    w[k] += deltaW[k];
                error = CalculateError(input, trainingOutput, w);
                if (error < error_min)
                    error_min = error;
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
            Vector<double>[] input = new Vector<double>[trainingInput.Length];
            //Agrego el valor 1 al principio del input.
            for (int i = 0; i < input.Length; i++)
            {
                input[i] = Vector<double>.Build.DenseOfEnumerable(new double[]{1}.Concat(trainingInput[i]));
            }
            int M = layers.Length;
            Matrix<double>[] w = new Matrix<double>[M];
            //Inicializo los pesos en valores aleatorios.
            for (int i = 0; i < M; i++)
            {
                w[i] = CreateMatrix.Random<double>(layers[i], i == 0 ? input[0].Count : layers[i - 1] + 1);
            }
            var w_min = w;
            Vector<double>[] V = new Vector<double>[M + 1];
            Vector<double>[] delta = new Vector<double>[M];
            Matrix<double>[] deltaW = new Matrix<double>[M];
            Vector<double>[] h = new Vector<double>[M];
            double error = 2 * minError + 1;
            double error_min = error;

            for(int i = 0; i < epochs && error_min > minError; i++)
            {


                int[] rand = Combinatorics.GeneratePermutation(input.Length);
                double lr = AdaptiveLearningRate ? optimizing(input, V, w, M, h, delta, trainingOutput, deltaW, batch, error, rand) : LearningRate;
                int j;
                for (j = 0; j < input.Length; j++)
                {
                    int index = rand[j];
                    V[0] = input[index];
                    for (int k = 0; k < M; k++)
                    {
                        h[k] = w[k] * V[k];
                        Vector<double> activationOutput = h[k].Map(g[k]);
                        //Agrego el valor 1 al principio de cada salida intermedia.
                        V[k + 1] = k + 1 < M ? Vector<double>.Build.DenseOfEnumerable(new double[] { 1 }.Concat(activationOutput)) : activationOutput;
                        
                    }
                    delta[M - 1] = h[M - 1].Map(gprime[M - 1]).PointwiseMultiply(trainingOutput[index] - V[M]);
                    
                    for (int k = M - 1; k > 0; k--)
                    {
                        var aux = w[k].TransposeThisAndMultiply(delta[k]);
                        //Salteo el primer valor ya que corresponde al delta de la neurona extra para el umbral (siempre tiene que tener activacion igual a 1).
                        aux = Vector<double>.Build.DenseOfEnumerable(aux.Skip(1));
                        delta[k - 1] = h[k - 1].Map(gprime[k - 1]).PointwiseMultiply(aux);

                    }

                    for (int k = 0; k < M; k++)
                    {
                        if (deltaW[k] != null)
                        {
                            deltaW[k] += lr * delta[k].OuterProduct(V[k]);
                            //deltaW[k] += LearningRate * delta[k].OuterProduct(V[k]);
                        }                           
                        else
                        {
                            deltaW[k] = lr * delta[k].OuterProduct(V[k]);
                            //deltaW[k] = LearningRate * delta[k].OuterProduct(V[k]);
                        }
                    }

                    if (j % batch == 0)
                    {
                        for (int k = 0; k < M; k++)
                            w[k] += deltaW[k];
                        error = CalculateError(input, trainingOutput, w);
                        if (error < error_min)
                        {
                            error_min = error;
                            w_min = w;
                        }
                        //Reinicio los deltaW para el proximo lote.
                        Array.Fill(deltaW, null);
                    }
                }
                if (j % batch != 0)
                {
                    for (int k = 0; k < M; k++)
                        w[k] += deltaW[k];
                    error = CalculateError(input, trainingOutput, w);
                    if (error < error_min)
                    {
                        error_min = error;
                        w_min = w;
                    }
                }

            }
            W = w_min;
        }
    }
}
