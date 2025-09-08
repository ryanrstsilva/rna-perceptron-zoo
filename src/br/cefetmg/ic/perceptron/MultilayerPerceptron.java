package br.cefetmg.ic.perceptron;

import java.util.Random;

public class MultilayerPerceptron implements NeuralNetwork {
    private int numInputs;
    private int numHidden;
    private int numOutputs;

    private double[][] weightsHidden;
    private double[][] weightsOutput;

    private double learningRate = 0.3;

    public MultilayerPerceptron(int numInputs, int numHidden, int numOutputs, double learningRate) {
        this(numInputs, numHidden, numOutputs);
        this.learningRate = learningRate;
    }

    public MultilayerPerceptron(int numInputs, int numHidden, int numOutputs) {
        this.numInputs = numInputs;
        this.numHidden = numHidden;
        this.numOutputs = numOutputs;
        weightsHidden = new double[this.numInputs + 1][numHidden];
        weightsOutput = new double[numHidden + 1][numOutputs];

        // Gerar pesos aleatórios no intervalo [-0.3, 0.3]
        Random rand = new Random();
        for (int i = 0; i < weightsHidden.length; i++) {
            for (int j = 0; j < weightsHidden[i].length; j++) {
                weightsHidden[i][j] = 0.6 * rand.nextDouble() - 0.3;
            }
        }
        
        for (int i = 0; i < weightsOutput.length; i++) {
            for (int j = 0; j < weightsOutput[i].length; j++) {
                weightsOutput[i][j] = 0.6 * rand.nextDouble() - 0.3;
            }
        }
    }

    public double[] trainOnSample(double[] sampleInput, double[] target) {
        double[][] forwardResult = forward(sampleInput);
        double[] input = forwardResult[0];
        double[] hidden = forwardResult[1];
        double[] output = forwardResult[2];

        // Compute output layer deltas
        double[] deltaOutput = new double[numOutputs];
        for (int i = 0; i < numOutputs; i++) {
            deltaOutput[i] = output[i] * (1 - output[i]) * (target[i] - output[i]);
        }

        // Compute hidden layer deltas
        double[] deltaHidden = new double[numHidden];
        for (int i = 0; i < numHidden; i++) {
            double sum = 0.0;
            for (int j = 0; j < numOutputs; j++) {
                sum += deltaOutput[j] * weightsOutput[i][j];
            }
            deltaHidden[i] = hidden[i] * (1 - hidden[i]) * sum;
        }

        // Atualização dos pesos da camada oculta.
        for (int i = 0; i < weightsHidden.length; i++) {
            for (int j = 0; j < weightsHidden[i].length; j++) {
                weightsHidden[i][j] += learningRate * deltaHidden[j] * input[i];
            }
        }

        // Atualização dos pesos da camada de saída.
        for (int i = 0; i < weightsOutput.length; i++) {
            for (int j = 0; j < weightsOutput[i].length; j++) {
                weightsOutput[i][j] += learningRate * deltaOutput[j] * hidden[i];
            }
        }

        return output;
    }

    public double[] executeOnSample(double[] sampleInput) {
        return forward(sampleInput)[2];
    }

    private double[][] forward(double[] sampleInput) {
        // Prepare input with bias
        double[] input = new double[sampleInput.length + 1];
        System.arraycopy(sampleInput, 0, input, 0, sampleInput.length);
        input[input.length - 1] = 1.0;

        // Hidden layer
        double[] hidden = new double[numHidden + 1];
        for (int i = 0; i < numHidden; i++) {
            double sum = 0.0;
            for (int j = 0; j < input.length; j++) {
                sum += input[j] * weightsHidden[j][i];
            }
            // Sigmoid activation
            hidden[i] = 1 / (1 + Math.exp(-sum));
        }
        hidden[numHidden] = 1.0; // Bias

        // Output layer
        double[] output = new double[numOutputs];
        for (int i = 0; i < numOutputs; i++) {
            double sum = 0.0;
            for (int j = 0; j < hidden.length; j++) {
                sum += hidden[j] * weightsOutput[j][i];
            }
            // Sigmoid activation
            output[i] = 1 / (1 + Math.exp(-sum));
        }
        return new double[][] { input, hidden, output };
    }
}
