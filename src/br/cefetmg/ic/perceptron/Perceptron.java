package br.cefetmg.ic.perceptron;

import java.util.Random;

public class Perceptron implements NeuralNetwork {
    private int numInputs;
    private int numOutputs;

    private double[][] weights;

    private double learningRate = 0.3;

    public Perceptron(int numInputs, int numOutputs, double learningRate) {
        this(numInputs, numOutputs);
        this.learningRate = learningRate;
    }

    public Perceptron(int numInputs, int numOutputs) {
        this.numInputs = numInputs;
        this.numOutputs = numOutputs;
        weights = new double[this.numInputs + 1][numOutputs];

        // Gerar pesos aleatórios no intervalo [-0.3, 0.3]
        Random rand = new Random();
        for (int i = 0; i < weights.length; i++) {
            for (int j = 0; j < weights[i].length; j++) {
                weights[i][j] = 0.6 * rand.nextDouble() - 0.3;
            }
        }
    }

    public double[] trainOnSample(double[] sampleInput, double[] target) {
        double[][] forwardResult = forward(sampleInput);
        double[] input = forwardResult[0];
        double[] output = forwardResult[1];

        // Compute deltas
        double[][] delta = new double[numOutputs][numInputs + 1];
        for (int i = 0; i < numOutputs; i++) {
            for (int j = 0; j < numInputs + 1; j++) {
                delta[i][j] = learningRate * (target[i] - output[i]) * input[j];
            }
        }

        // Update weights
        for (int i = 0; i < numOutputs; i++) {
            for (int j = 0; j < numInputs + 1; j++) {
                weights[j][i] += delta[i][j];
            }
        }

        return output;
    }

    public double[] executeOnSample(double[] input) {
        return forward(input)[1];
    }

    private double[][] forward(double[] sampleInput) {
        // Prepare input with bias
        double[] input = new double[sampleInput.length + 1];
        System.arraycopy(sampleInput, 0, input, 0, sampleInput.length);
        input[input.length - 1] = 1.0;

        // Forward pass
        double[] sum = new double[numOutputs];
        double[] output = new double[numOutputs];
        for (int i = 0; i < numOutputs; i++) {
            for (int j = 0; j < input.length; j++) {
                sum[i] += input[j] * weights[j][i];
            }
            output[i] = 1 / (1 + Math.exp(-sum[i]));
        }

        return new double[][] { input, output };
    }
}
