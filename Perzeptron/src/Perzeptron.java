import java.io.*;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.Random;

/**
 * Created by aw3s0_000 on 27.10.2015.
 */
public class Perzeptron {
    private int N; //number of properties from file
    private int M; //number of neurons from file
    private int K; //number of objects from file
    private static final double LEARNING_RATE = 0.01;
    private static final int ITERATIONS_NUM = 200;
    private double[][] inputs;
    private double[][] teacher; //matrix of teacher values
    private double[][] weights; //matrix of weights
    private double[][] resultsMult; //result after multiplication of inputs on weights matrixes
    private double[][] sigmResult; //result after sigmoid function

    public int getNumberOfNeurons() {
        return M;
    }

    public int getNumberOfProperties() {
        return N;
    }

    public int getNumberOfObjects() {
        return K;
    }

    public double[][] getResultsMult() {
        return resultsMult;
    }

    public double[][] getSigmResult() {
        return sigmResult;
    }

    public double[][] getInputs() {
        return inputs;
    }

    public double[][] getTeacher() {
        return teacher;
    }

    public double[][] getWeights() {
        return weights;
    }

    public double sigmoid(double val) {
        return (1/( 1 + Math.pow(Math.E,(-1*val))));
    }

    public void calcSigm() {
        sigmResult = new double[K][M];
        for (int i = 0; i < K; i++) {
            for (int j = 0; j < M; j++) {
                sigmResult[i][j] = sigmoid(resultsMult[i][j]);
            }
        }
    }

    /**
     * Function that creates 2-D array of inputs from file
     * File structure:
     * 1) M - number of neurons - first line
     * 2) K - number of objects (,) N - number of properties
     * @param filename
     * @throws Exception
     */
    public Perzeptron(String filename) throws IOException {
        InputStream stream = ClassLoader.getSystemResourceAsStream(filename);
        BufferedReader buffer = new BufferedReader(new InputStreamReader(stream));
        String line;
        LinkedList<ArrayList<Double>> tempInpList = new LinkedList<>();
        LinkedList<ArrayList<Double>> tempTeacherList = new LinkedList<>();

        while ((line = buffer.readLine()) != null) {
            String[] vals = line.trim().split("\\t"); //to parse

            String[] inpVals = vals[0].split(" ");
            String[] teacherVals = vals[1].split(" ");
            ArrayList<Double> inputRow = new ArrayList<>();
            ArrayList<Double> teacherRow = new ArrayList<>();

            for (int i = 0; i < inpVals.length; i++) {
                inputRow.add(Double.parseDouble(inpVals[i]));
            }

            for (int i = 0; i < teacherVals.length; i++) {
                teacherRow.add(Double.parseDouble(teacherVals[i]));
            }

            tempInpList.add(inputRow);
            tempTeacherList.add(teacherRow);
        }

        K = tempInpList.size();
        N = tempInpList.get(0).size();
        M = tempTeacherList.get(0).size();

        inputs = new double[K][N + 1];
        teacher = new double[K][M];

        for (int i = 0; i < K; i++) {
            ArrayList<Double> list = tempInpList.get(i);
            for (int j = 0; j < N; j++) {
                inputs[i][j] = list.get(j);
            }
        }

        //Initialization of bias row (each element equals to 1
        for (int i = 0; i < K; i++) {
            inputs[i][N] = 1;
        }

        for (int i = 0; i < K; i++) {
            ArrayList<Double> list = tempTeacherList.get(i);
            for (int j = 0; j < M; j++) {
                teacher[i][j] = list.get(j);
            }
        }

    }

    /**
     * function to calculate gradient descent
     */
    private void gradientStep() {
        double[][] transposedInputs = Matrix.transpose(inputs); //need to transpose input matrix, because it doesn't match dimensions
        double[][] errorResult = new double[K][M]; //matrix to store intermediate result

        for (int i = 0; i < K; i++) {
            for (int j = 0; j < M; j++) {
                errorResult[i][j] = LEARNING_RATE *
                        (teacher[i][j] - sigmResult[i][j]) *
                        (1 - sigmResult[i][j]) *
                        sigmResult[i][j];
            }
        }
        //get new weights
        weights = Matrix.add(weights, Matrix.multiply(transposedInputs, errorResult));
    }

    /**
     * To generate weight matrix
     * @param minWeight
     * @param maxWeight
     * @return
     */
    public void initWeights(double minWeight, double maxWeight) {
        weights = new double[N+1][M];
        Random r = new Random();

        for (int i = 0; i < weights.length; i++) {
            for (int j = 0; j < weights[i].length; j++) {
                weights[i][j] = minWeight + (maxWeight - minWeight) * r.nextDouble();
            }
        }
    }

    public void multiplyWeightsAndInputs() {
        resultsMult = Matrix.multiply(inputs, weights);
    }

    public double[][] getError() {
        return Matrix.subtract(teacher, sigmResult);
    }

    public static void main(String[] args) throws Exception {
        double minWeight = -0.5;
        double maxWeight = 0.5;

        Perzeptron perzeptron = new Perzeptron("PA-A-train.dat");

        System.out.println("Number of objects: " + perzeptron.getNumberOfObjects() + "\n");
        System.out.println("Number of neurons: " + perzeptron.getNumberOfNeurons() + "\n");
        System.out.println("Number of properties: " + perzeptron.getNumberOfProperties() + "\n");

        Utils.printMatrix("Input matrix: ", perzeptron.getInputs());
        Utils.printMatrix("Teacher matrix: ", perzeptron.getTeacher());

        perzeptron.initWeights(minWeight, maxWeight);
        Utils.printMatrix("Weight matrix: ", perzeptron.getWeights());

        for (int i = 0; i < ITERATIONS_NUM; i++) {
            perzeptron.multiplyWeightsAndInputs();
            perzeptron.calcSigm();
            perzeptron.gradientStep();
        }

        Utils.printMatrix("Sigmoid result: ", perzeptron.getSigmResult());
        Utils.printMatrix("New weights: ", perzeptron.getWeights());
        Utils.printMatrix("Error matrix: ", perzeptron.getError());
    }
}
