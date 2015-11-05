import java.io.*;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.Random;

/**
 * Created by aw3s0_000 on 04.11.2015.
 * Aleksandr Korovin and Andrei Zhukov. Uni Bonn. Neural nets assignment PB-A.
 */
public class MLP
{
    public class MLPLayer
    {
        private int nOutput;
        private int nInput;
        private ITransferFunction transferFunction;
        private double[][] input;
        private double[][] weights;
        private double[][] dweights;
        private double[][] output;

        public int getInputNumber() {
            return nInput;
        }

        public MLPLayer(int inputSize, int outputSize, ITransferFunction transferFunction, Random random) {
            this.transferFunction = transferFunction;
            this.nOutput = outputSize;
            this.nInput = inputSize;
            initializeWeights(random);
        }

        private double[][] applyBias(double[][] input, int nInput) {
            double[][] newInput = new double[input.length][nInput + 1];

            for (int i = 0; i < input.length; i++) {
                for (int j = 0; j < input[i].length; j++) {
                    newInput[i][j] = input[i][j];
                }
            }

            //Initialization of bias row (each element equals to 1
            for (int i = 0; i < input.length; i++) {
                newInput[i][nInput] = 1;
            }

            return newInput;
        }

        /**
         * get output of MLPLayer
         * @param input
         */
        public double[][] run(double[][] input) {
            if (input == null)
            {
                throw new IllegalArgumentException("input matrix cant be null");
            }
            int nPattern = input.length;
            double[][] newInput = applyBias(input, nInput);
            double[][] multRes = Matrix.multiply(newInput, weights);

//            output = new double[nPattern + 1][nOutput];
            output = new double[nPattern][nOutput];

            for (int i = 0; i < output.length; i++) {
                for (int j = 0; j < output[i].length; j++) {
                    output[i][j] = transferFunction.calculate(multRes[i][j]);
                }
            }

            return output;
        }

        private void initializeWeights(Random random) {
            weights = new double[nInput + 1][nOutput];
            dweights = new double[nInput + 1][nOutput];

            for (int i = 0; i < weights.length; i++) {
                for (int j = 0; j < weights[i].length; j++) {
                    weights[i][j] = MLP.MIN_GEN_WEIGHT + (MLP.MAX_GEN_WEIGHT - MLP.MIN_GEN_WEIGHT) * random.nextDouble();
                }
            }
        }
    }

    private static final double LEARNING_RATE = 0.01;
    private static final int ITERATIONS_NUM = 200;
    //GENERATION CONSTANTS
    private static final double MIN_GEN_WEIGHT = -2;
    private static final double MAX_GEN_WEIGHT = 2;
    private static final int GEN_SEED = 2;
    private final Random RANDOMIZER = new Random();
    //LAYERS
    private static final int HID_LAYER_NUM_1 = 6; //in case of 1-hidden-layer and 2-hidden layers.
    private static final int HID_LAYER_NUM_2 = 8; //in case of 2 hidden layers.
    private ArrayList<MLPLayer> layers = new ArrayList<>();
    //STI = Sigmoid (0 index), Tanh (1 index), Identity (2 index)
    private static final ArrayList<ITransferFunction> TRANSFER_FUNCTIONS_STI = new ArrayList<ITransferFunction>(){{
        add(new SigmoidFunction()); //index == 0
        add(new TanhFunction()); //index == 1
        add(new IdentityFunction()); //index == 2
    }};
    //MATRICES
    private double[][] fileInputs;
    private double[][] teacher; //matrix of teacher values
    private double[][] output; //result after sigmoid function

    public double[][] getFileInputs() {
        return fileInputs;
    }

    public double[][] getTeacher() {
        return teacher;
    }

    private void setRandomizer() {
        RANDOMIZER.setSeed(GEN_SEED);
    }

    /**
     * Function that creates 2-D array of fileInputs from file
     * File structure:
     * 1) nOutput - number of neurons - first line
     * 2) nPatterns - number of objects (,) nProperties - number of properties
     * @param filename
     * @param hiddenLayerSizes
     * @throws Exception
     */
    public MLP(String filename, int[] hiddenLayerSizes) throws IOException {
        InputStream stream = ClassLoader.getSystemResourceAsStream(filename);
        BufferedReader buffer = new BufferedReader(new InputStreamReader(stream));
        String line;
        LinkedList<ArrayList<Double>> tempInpList = new LinkedList<>();
        LinkedList<ArrayList<Double>> tempTeacherList = new LinkedList<>();

        while ((line = buffer.readLine()) != null) {
            if (line.contains("#")) {
                continue;
            }
            String[] vals = line.trim().split("    "); //to parse

            String[] inpVals = vals[0].split(" ");
            String[] teacherVals = vals[1].split(" ");
            ArrayList<Double> inputRow = new ArrayList<>();
            ArrayList<Double> teacherRow = new ArrayList<>();

            for (int i = 0; i < inpVals.length; i++) {
                if (!inpVals[i].isEmpty())
                    inputRow.add(Double.parseDouble(inpVals[i]));
            }

            for (int i = 0; i < teacherVals.length; i++) {
                if (!teacherVals[i].isEmpty())
                    teacherRow.add(Double.parseDouble(teacherVals[i]));
            }

            tempInpList.add(inputRow);
            tempTeacherList.add(teacherRow);
        }

        int nPatterns = tempInpList.size();
        int nProperties = tempInpList.get(0).size();
        int nOutput = tempTeacherList.get(0).size();

        fileInputs = new double[nPatterns][nProperties];
        teacher = new double[nPatterns][nOutput];

        for (int i = 0; i < nPatterns; i++) {
            ArrayList<Double> list = tempInpList.get(i);
            for (int j = 0; j < nProperties; j++) {
                fileInputs[i][j] = list.get(j);
            }
        }

        for (int i = 0; i < nPatterns; i++) {
            ArrayList<Double> list = tempTeacherList.get(i);
            for (int j = 0; j < nOutput; j++) {
                teacher[i][j] = list.get(j);
            }
        }

        ArrayList<Integer> layerSizes = setLayerSizes(nProperties, nOutput, hiddenLayerSizes);

        initializeLayers(layerSizes);
    }

    private ArrayList<Integer> setLayerSizes(int nInput, int nOutput, int[] hiddenLayerSizes) {
        ArrayList<Integer> layerSizes = new ArrayList<>();
        layerSizes.add(nInput);

        if (hiddenLayerSizes != null || hiddenLayerSizes.length != 0) {
            for (int i = 0; i < hiddenLayerSizes.length; i++) {
                layerSizes.add(hiddenLayerSizes[i]);
            }
        }

        layerSizes.add(nOutput);
        return layerSizes;
    }

    private void initializeLayers(ArrayList<Integer> layerSizes) {
        if (layerSizes.size() > 4) {
            throw new IllegalArgumentException("Number of layers shouldnt be more than 4");
        }
        setRandomizer();
        for (int i = 1; i < layerSizes.size(); i++) {
            layers.add(new MLPLayer(layerSizes.get(i-1),
                    layerSizes.get(i), TRANSFER_FUNCTIONS_STI.get(i - 1), RANDOMIZER));
        }
    }

    private void iteration() {
        double[][] output = fileInputs;
        for (int i = 0; i < ITERATIONS_NUM; i++) {
            for (MLPLayer layer : layers) {
                output = layer.run(output);
            }
            backPropagation();
        }
    }

    /**
     * function to calculate gradient descent
     */
    private void backPropagation() {
//        double[][] transposedInputs = Matrix.transpose(fileInputs); //need to transpose input matrix, because it doesn't match dimensions
//        double[][] errorResult = new double[nPatterns][nOutput]; //matrix to store intermediate result
//
//        for (int i = 0; i < nPatterns; i++) {
//            for (int j = 0; j < nOutput; j++) {
//                errorResult[i][j] = LEARNING_RATE *
//                        (teacher[i][j] - sigmResult[i][j]) *
//                        (1 - sigmResult[i][j]) *
//                        sigmResult[i][j];
//            }
//        }
//        //get new weights
//        weights = Matrix.add(weights, Matrix.multiply(transposedInputs, errorResult));
    }

    public static void main(String[] args) throws Exception {
        MLP perzeptron = new MLP("training.dat", new int[]{HID_LAYER_NUM_1, HID_LAYER_NUM_2});
        Utils.printMatrix("Input matrix: ", perzeptron.getFileInputs());
        Utils.printMatrix("Teacher matrix: ", perzeptron.getTeacher());

        perzeptron.iteration();
    }

    /**
     * Created by aw3s0_000 on 03.11.2015.
     */
    public interface ITransferFunction {
        double calculate(double val);
    }

    public static class SigmoidFunction implements ITransferFunction {
        @Override
        public double calculate(double val) {
            return (1/( 1 + Math.pow(Math.E,(-1*val))));
        }
    }

    public static class TanhFunction implements ITransferFunction {
        @Override
        public double calculate(double val) {
            return Math.tanh(val);
        }
    }

    public static class IdentityFunction implements ITransferFunction {
        @Override
        public double calculate(double val) {
            return val;
        }
    }
}


