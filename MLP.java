package production;
import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;

/**
 * This program is learningRate machine learning system to categorise one of the UCI digit tasks.
 * It has two-fold test of given data sets
 * 
 * Algorithm: Multi Layer Perceptron (Input-Hidden-Output)
 *
 * @author  M.Hammad Khokhar
 * @version 2.0
 * @since   14-03-2019
 */
public class MLP {

	//Test and Train Data/Index Variables
	private static double[][] trainData = new double[2810][65];
	private static double[][] testData = new double[2810][65];

	//Score calculation variables
	private static double foldScore = 0;
	private static double firstFoldScore = 0;
	private static double secondFoldScore = 0;
	private static double foldAverage = 0;

	//Hidden layer numbers
	private static int numOfHiddenLayers=100;
	private static int numOfOutput=10;

	//Hidden Layer Array
	private static double [] outputHidden = new double [numOfHiddenLayers];
	//Output Layer Output Array
	private static double [] outputOutput = new double [numOfOutput];
	//Hidden nodes delta array
	private static double [] deltaHiddenLayer = new double [numOfHiddenLayers];
	//Output nodes delta array
	private static double [] deltaOutputNodes = new double [numOfOutput];

	//total input rows
	private static int numOfInput=64;
	
	//hidden and output layer number of weights variables
	private static int numOfHiddenLayerWeights = numOfInput+1;
	private static int numOfOutputWeights = numOfHiddenLayers+1;

	//hidden weights array
	private static double [][] hiddenLayerWeights = new double [numOfHiddenLayers][numOfHiddenLayerWeights];
	private static double [][] outputWeights = new double [numOfOutput][numOfOutputWeights];

	//confusion matrix array
	private static int [][]confusionMatrix = new int [10][10];

	//default learning rate
	private static double learningRate = 0.09;
	//number of test to perform two-fold
	private static int testNum=0;
	
	/**
	 * This function is unit activation
	 * @param learningRateInput default 0.09
	 * @return activated neurons
	 */
	private static double activeNeuralNetwork(double learningRateInput) {
		double unitActivation;
		unitActivation= 1 + Math.exp(-learningRateInput);
		return 1/unitActivation;
	}

	/**
	 * This function calculates hidden-layer output
	 */
	private static void calculateHiddenLayerOutput(double[] image) {
		double sumHL;
		for (int i = 0; i < numOfHiddenLayers; i++) {
			sumHL = 0;
			for (int k =   0; k <  numOfHiddenLayerWeights-1; k++ ){
				sumHL +=image[k]*hiddenLayerWeights[i][k];
			}
			sumHL += hiddenLayerWeights[i][numOfHiddenLayerWeights -1 ];
			outputHidden[i] = activeNeuralNetwork(sumHL);
		}
	}

	/**
	 * This function calculates output-layer output
	 */
	private static void calculateOutputLayerOutput() {
		double sumOL;
		for (int i = 0; i < numOfOutput; i++) {
			sumOL = 0;
			for (int k =   0; k <  numOfOutputWeights-1; k++ ) {
				sumOL += outputHidden[k]*outputWeights[i][k];
			}
			sumOL += outputWeights[i][numOfOutputWeights -1];
			outputOutput[i] = activeNeuralNetwork(sumOL);
		}
	}


	/**
	 * This function calculates output-layer delta nodes
	 */
	private static void calculateOutputLayerDeltas(double digit) {
		double target;
		for (int i = 0; i < outputOutput.length;i++) {
			if(i == digit) {
				target=1;
			}
			else {
				target=0;
			}
			deltaOutputNodes[i]= outputOutput[i]*(1-outputOutput[i])*(outputOutput[i]-target);
		}
	}

	/**
	 * This function calculates hidden-layer delta nodes
	 */
	private static void calculateHiddenLayerDeltas() {
		//double sum=0;
		for (int i = 0; i < numOfHiddenLayers; i++) {
			deltaHiddenLayer[i] =0;
			for (int k =   0; k <  numOfOutput; k++ ) {
				deltaHiddenLayer[i] +=	((deltaOutputNodes[k] * (outputWeights[k][i])));
			}
			deltaHiddenLayer[i]*=(1 - outputHidden[i]) * outputHidden[i];
		}
	}

	/**
	 * This function updates weight between output layer and hidden layer nodes. 
	 */
	private static void updateOutputAndHiddenLayerWeights() {
		for (int i =0; i < numOfOutput; i++) {
			for (int k =   0; k <  numOfOutputWeights-1; k++ ) {
				outputWeights[i][k] += -learningRate *  deltaOutputNodes[i] * outputHidden[k];
			}
			// Bias(Offset) - It is an extra input to neurons and it is always 1, and has it's own connection weight.
			outputWeights[i][numOfOutputWeights-1] += -learningRate * deltaOutputNodes[i];
		}
	}

	/**
	 * This function is for hidden layer weight update.
	 * @param input image
	 */
	private static void updateHiddenLayerWeights(double[] input) {
		for (int i =0; i < numOfHiddenLayers; i++) {
			for (int k =   0; k <  numOfHiddenLayerWeights-1; k++ ) {
				hiddenLayerWeights[i][k] += -learningRate *  deltaHiddenLayer[i] * input[k];
			}
			//new weight - update
			hiddenLayerWeights[i][numOfHiddenLayerWeights-1] += -learningRate *  deltaHiddenLayer[i];
		}
	}


	/**
	 * Generate random weights for epoch
	 */
	private static void randomWeights() {
		double max = 0.5;
		// Range for the random weights.
		double min = -0.5;
		for (int hiddenLayer = 0; hiddenLayer < numOfHiddenLayers; hiddenLayer++) {
			for (int inputLayer = 0; inputLayer < numOfHiddenLayerWeights; inputLayer++) {
				hiddenLayerWeights[hiddenLayer][inputLayer]= min +(Math.random()*(max - min));
			}
		}
		for ( int OutputLayer=0;  OutputLayer<numOfOutput; OutputLayer++) {
			for (int outputWeight=0; outputWeight<numOfOutputWeights; outputWeight++ ) {
				outputWeights[OutputLayer][outputWeight]= min +(Math.random()*(max - min));
			}
		}
	}

	/**
	 *  Train the network weights for the Perceptron.
	 */
	private static void trainPerceptrons() {
		randomWeights();
		for (int i = 0; i < 500; i++){
			for(int eachimage=0; eachimage<2810; eachimage++) {
				calculateHiddenLayerOutput(trainData[eachimage]);
				calculateOutputLayerOutput();
				calculateOutputLayerDeltas(trainData[eachimage][64]);
				calculateHiddenLayerDeltas();
				updateOutputAndHiddenLayerWeights();
				updateHiddenLayerWeights(trainData[eachimage]);
			}
		}
	}


	/**
	 * Two-Fold test function
	 */
	private static void testPerceptrons() {
		int outputNode = 0;
		for (int i =0; i < 2810; i++) {
			calculateHiddenLayerOutput(testData[i]);
			calculateOutputLayerOutput();
			for (int k =   0; k <  numOfOutput; k++ ) {
				if (outputOutput[k] > 0.5){
					outputNode=k; // finding the large number
				}
			}
			if (outputNode==testData[i][64]) {
				foldScore++; // its for both test. This contain the number of correct classification for each test.
			}
			// confusion confusionMatrix
			confusionMatrix[outputNode][(int) testData[i][64]]++;
		}
		System.out.println("No of correct digit: " + (int)foldScore);
		foldAverage += foldScore;

		// Generate confusion matrix
		//First loop = rows and second = columns
		//System.out.println("Confusion Matrix Below:");
		for (int[] confusionMatrix : confusionMatrix) {
			for (int confusionMatrix1 : confusionMatrix) {
				//enable to print confusion matrix
				//System.out.print( confusionMatrix1 + "      " );
			}
			//System.out.println();
		}

		if(testNum==0) {
			firstFoldScore += foldScore;
			foldScore=0;
		}
		else {
			secondFoldScore += foldScore;

			System.out.println(" *-----------------------------------------------*");
			System.out.println(" | First Fold Percentage: "+ firstFoldScore * 100 / 2810 +"    |");
			System.out.println(" | Second Fold Percentage: "+ secondFoldScore * 100 / 2810 +"   |");
			System.out.println(" *| Average Folds Percentage: "+ foldAverage * 100 / 5620 +" |* ");
			System.out.println(" x-----------------------------------------------x");

		}
	}

	/**
	 * This function reads data sets and runs addLineToData
	 * @param testNum default 0 ends with 1 in total 2 tests will be performed
	 */
	private static void ReadDataSets(int testNum) throws Exception{
		String trainPath, testPath;
		if(testNum==0) {
			trainPath = "C:\\Users\\Defender\\IdeaProjects\\AI_Coursework_2\\src\\cw2DataSet2.csv";
			testPath = "C:\\Users\\Defender\\IdeaProjects\\AI_Coursework_2\\src\\cw2DataSet1.csv";
		}
		else {
			trainPath = "C:\\Users\\Defender\\IdeaProjects\\AI_Coursework_2\\src\\cw2DataSet1.csv";
			testPath = "C:\\Users\\Defender\\IdeaProjects\\AI_Coursework_2\\src\\cw2DataSet2.csv";
		}

		addLineToData(trainPath, trainData);
		addLineToData(testPath, testData);

		System.out.println("----------------------------------------------------------");
		System.out.println("Training and Testing Perceptrons. This might take a while. Fold/Test Number:" + testNum);
		System.out.println("----------------------------------------------------------");
	}

	/**
	 * This function adds line to data
	 * @param fileName Testing Data
	 * @param trainData Training Data
	 * @throws Exception if file not found
	 */
	private static void addLineToData(String fileName, double[][] trainData) throws Exception{
		int theLine = 0;
		try {
			BufferedReader reader = new BufferedReader(new FileReader(fileName));
			String line;
			while ((line = reader.readLine()) != null) {
				String[] data = line.split(",");
				for (int i = 0; i < 65; i++) {
					trainData[theLine][i] = Integer.parseInt(data[i]);
				}
				theLine++;
			}
			reader.close();
		}
		//Throws Exception If File Not Found
		catch (FileNotFoundException ex) {
			ex.printStackTrace();
		}
	}

	public static void main(String[] args) throws Exception{
		// Current time
		long start = System.currentTimeMillis();
		for (int i = 0; i <5; i++) {
			Thread.sleep(60);
		}
		//Run Two-Fold test loop
		for(testNum = 0; testNum <= 1; testNum++){
			//Execute functions below
			ReadDataSets(testNum);
			trainPerceptrons();
			testPerceptrons();
		}
		// finding the time after the functions has executed
		long end = System.currentTimeMillis();
		//finding the time difference and converting it into seconds
		float sec = (end - start) / 1000F; System.out.println("Time taken in execution "+ sec + " seconds");
	}
}
	
