import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Created by aw3s0_000 on 06.11.2015.
 */
public class FileInput
{
    private int nPatterns;
    private int nOutput;
    private int nInput;
    private double[][] fileInputs;
    private double[][] teacher; //matrix of teacher values

    public int getnPatterns() {
        return nPatterns;
    }

    public int getnOutput() {
        return nOutput;
    }

    public int getnInput() {
        return nInput;
    }

    public double[][] getFileInputs() {
        return fileInputs;
    }

    public double[][] getTeacher() {
        return teacher;
    }

    public int getSizeByVariableName(String varName, String line) {
        String ptrn = String.format("%s=(.+?)\\s+", varName);
        Pattern regex = Pattern.compile(ptrn);
        Matcher matcher = regex.matcher(line);
        matcher.find();
        String str = matcher.group(1);
        return Integer.parseInt(str);
    }

    public FileInput(String filename) throws IOException {
        InputStream stream = ClassLoader.getSystemResourceAsStream(filename);
        BufferedReader buffer = new BufferedReader(new InputStreamReader(stream));
        String line;
        LinkedList<ArrayList<Double>> tempInpList = new LinkedList<>();
        LinkedList<ArrayList<Double>> tempTeacherList = new LinkedList<>();
        int commentIndex = 0;
        int P = 0;
        int N = 0;
        int M = 0;
        while ((line = buffer.readLine()) != null) {
            if (line.contains("#")) {
                if (commentIndex == 1) {
                    P = getSizeByVariableName("P", line);
                    N = getSizeByVariableName("N", line);
                    M = getSizeByVariableName("M", line);
                }
                commentIndex++;
                continue;
            }

            if(line == null || line.isEmpty()) break;
            String[] valsWithEmpties = line.trim().split(" "); //to parse
            ArrayList<String> valsWithoutEmpties = new ArrayList<>();

            ArrayList<Double> inputRow = new ArrayList<>();
            ArrayList<Double> teacherRow = new ArrayList<>();

            for (String val : valsWithEmpties) {
                if (val.isEmpty()) continue;
                else {
                    valsWithoutEmpties.add(val);
                }
            }

            for (int i = 0; i < N + M; i++) {
                //first N elements are input. then teacher
                if (i < N) {
                    inputRow.add(Double.parseDouble(valsWithoutEmpties.get(i)));
                }
                else {
                    teacherRow.add(Double.parseDouble(valsWithoutEmpties.get(i)));
                }
            }

            tempInpList.add(inputRow);
            tempTeacherList.add(teacherRow);
        }

        nPatterns = tempInpList.size();
        nInput = tempInpList.get(0).size();
        nOutput = tempTeacherList.get(0).size();

        fileInputs = new double[nPatterns][nInput];
        teacher = new double[nPatterns][nOutput];

        for (int i = 0; i < nPatterns; i++) {
            ArrayList<Double> list = tempInpList.get(i);
            for (int j = 0; j < nInput; j++) {
                fileInputs[i][j] = list.get(j);
            }
        }

        for (int i = 0; i < nPatterns; i++) {
            ArrayList<Double> list = tempTeacherList.get(i);
            for (int j = 0; j < nOutput; j++) {
                teacher[i][j] = list.get(j);
            }
        }
    }
}
