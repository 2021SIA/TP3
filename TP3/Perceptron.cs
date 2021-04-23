using MathNet.Numerics.LinearAlgebra;
using System;
using System.Collections.Generic;
using System.Text;

namespace TP3
{
    interface Perceptron
    {
        Vector<double> Map(Vector<double> input);
    }
}
