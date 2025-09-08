package br.cefetmg.ic.perceptron;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class Dataset {
    private List<Sample> samples;

    public Dataset(String filePath) {
        samples = loadDataset(filePath);
    }

    public static List<Sample> loadDataset(String filePath) {
        List<Sample> dataset = new ArrayList<>();
        
        try (BufferedReader br = new BufferedReader(new FileReader(filePath))) {
            String line;
            while ((line = br.readLine()) != null) {
                String[] values = line.split(",");

                double[] inputs = new double[16];
                for (int i = 0; i < 12; i++) inputs[i] = Double.parseDouble(values[i + 1]);
                inputs[12] = Double.parseDouble(values[13]) / 8.0;
                for (int i = 13; i < 16; i++) inputs[i] = Double.parseDouble(values[i + 1]);

                int classType = Integer.parseInt(values[17]);
                double[] outputs = new double[7];

                if (classType >= 1 && classType <= 7) {
                    outputs[classType - 1] = 1;
                }

                dataset.add(new Sample(inputs, outputs));
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

        return dataset;
    }

    public List<Sample> getSamples() {
        return samples;
    }

    static class Sample {
        private double[] input;
        private double[] output;

        Sample(double[] input, double[] output) {
            this.input = input;
            this.output = output;
        }

        public double[] getInput() {
            return input;
        }

        public double[] getOutput() {
            return output;
        }
    }
}
