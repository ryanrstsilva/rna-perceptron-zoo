package br.cefetmg.ic.perceptron;

public class Runner {
    public static void main(String[] args) {
        Dataset zoo = new Dataset("./data/zoo.data");
        NeuralNetwork perceptron = new Perceptron(16, 7, 0.3);
        NeuralNetwork mlp = new MultilayerPerceptron(16, 14, 7, 0.3);

        executeTraining(zoo, perceptron, 1000);
        executeTraining(zoo, mlp, 10000);
    }

    public static void executeTraining(Dataset dataset, NeuralNetwork ann, int numEpochs) {
        double prevEpochError = Double.MAX_VALUE;
        double errorThreshold = 0.01; // Stop when error is below this value
        double minErrorChange = 0.0001; // Stop when error change is below this value

        for (int i = 0; i < numEpochs; i++) {
            double epochApproximationError = 0;
            double epochClassificationError = 0;

            for (Dataset.Sample sample : dataset.getSamples()) {
                double[] x_in = sample.getInput();
                double[] y = sample.getOutput();

                double[] o = ann.trainOnSample(x_in, y);

                double sampleApproximationError = 0;
                for (int j = 0; j < o.length; j++) {
                    sampleApproximationError += Math.abs(y[j] - o[j]);
                }
                epochApproximationError += sampleApproximationError;

                double[] o_t = new double[o.length];
                for (int j = 0; j < o_t.length; j++) {
                    o_t[j] = (o[j] >= 0.5) ? 1 : 0;
                }

                double difference = 0;
                for (int j = 0; j < y.length; j++) {
                    difference += Math.abs(y[j] - o_t[j]);
                }

                double sampleClassificationError = (difference > 0) ? 1 : 0;
                epochClassificationError += sampleClassificationError;
            }

            // Calculate normalized errors
            double normalizedApproxError = epochApproximationError / dataset.getSamples().size();
            double normalizedClassError = epochClassificationError / dataset.getSamples().size();

            System.out.println(String.format("Epoch %d - E_approx: %.4f - E_class: %.4f", i,
                    normalizedApproxError,
                    normalizedClassError));

            // Check stopping conditions
            if (normalizedApproxError < errorThreshold) {
                System.out.println("Parada: Erro de aproximação abaixo do limite.");
                break;
            }

            double errorChange = Math.abs(prevEpochError - normalizedApproxError);
            if (errorChange < minErrorChange && i > 100) { // Garantir algum treinamento mínimo.
                System.out.println("Parada: Convergência detectada.");
                break;
            }

            prevEpochError = normalizedApproxError;
        }
    }
}
