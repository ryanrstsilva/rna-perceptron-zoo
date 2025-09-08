package br.cefetmg.ic.perceptron;
public interface NeuralNetwork {
    public double[] trainOnSample(double[] input, double[] target);
    public double[] executeOnSample(double[] input);
}
