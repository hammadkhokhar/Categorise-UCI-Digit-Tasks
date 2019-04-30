package production;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.Scanner;

/**
 * This program is learningRate machine learning system to categorise one of the UCI digit tasks.
 * It has two-fold test of given data sets
 *
 * @author  M.Hammad Khokhar
 * @version 1.0
 * @since   15-02-2019
 */

public class Baseline {
    
    /**
     * Global variables
     */
    private static int [][] dataSet = new int[65][5620];
    private static double firstFoldScore = 0;
    private static double secondFoldScore = 0;
    private static double foldAverage = 0;

    /**
     * Distance commuter
     * @param row1 dataSet 1 row
     * @param row2 dataSet 2 row
     * @return Distance
     */
    private static double getDistance(int row1, int row2) {
        double distance = 0;
        for(int col=0; col<64; col++){
            int cellDiff = dataSet[col][row1] - dataSet[col][row2];
            distance = distance + (cellDiff*cellDiff);
        }
        return Math.sqrt( distance );
    }

    /**
     *
     * @param item which one to compare
     * @param startTrain where to start
     * @param endTrain where to end
     * @return the nearest distance
     *
     */
    private static int categorizeRow(int item, int startTrain, int endTrain){
        int nearestItem = -1;
        double nearestDistance = Double.MAX_VALUE;
        for(int trainRow = startTrain; trainRow < endTrain; trainRow++){
            // Get the distance
            double distance = getDistance( item, trainRow );
            // Finding the nearest Distance
            if(distance < nearestDistance){
                nearestDistance = distance;
                nearestItem = trainRow;
                //print the nearest distance
                //System.out.println( nearestDistance );
            }
        }
        return dataSet[64][nearestItem];

    }

    /**
     * TWO-FOLD TESTING
     * testRow is from which row you want to start test;
     * F-F : First Fold
     * S-F: Second Fold
     */
    private static void categorise(){
        //This runs training from 2810 row to 5619 and test from 0 to 2810
        System.out.println( "**------------------First Fold--------------------**" );
        for(int testRow =0; testRow < 2810; testRow ++){
            int Result = categorizeRow( testRow, 2810, 5619 );
            //Print the result and predicted value
            //System.out.println( "F-F| Result-" + Result + " | Predicted-" + dataSet[64][testRow] + " |" );
            if (Result == dataSet[64][testRow]){
                //increase second fold score
                firstFoldScore++;
                //increase average fold score
                foldAverage++;
            }
        }
        System.out.println(" X --------  Total Score for first fold | " + (int)firstFoldScore + " | Out of 2810 -------- X ");

        //This runs training from 0 to 2808 and test from 2810 to 5620
        System.out.println( "**------------------Second Fold--------------------**" );
        for (int testRow = 2810; testRow < 5620; testRow ++){
            int Result = categorizeRow( testRow, 0, 2808 );
            //Print the result and predicted value
            //System.out.println( "S-F| Result-" + Result + " | Predicted-" + dataSet[64][testRow] + " |" );
            if(Result == dataSet[64][testRow]){
                //increase second fold score
                secondFoldScore++;
                //increase average fold score
                foldAverage++;
            }
        }
        System.out.println(" X --------  Total Score for second fold | " + (int)secondFoldScore + " | Out of 2810 -------- X ");


        System.out.println(" -----------------------------------------------");
        System.out.println(" | First Fold Percentage: "+ firstFoldScore * 100 / 2810 +"    |");
        System.out.println(" | Second Fold Percentage: "+ secondFoldScore * 100 / 2810 +"   |");
        System.out.println(" | Average Folds Percentage: "+ foldAverage * 100 / 5620 +" |");
        System.out.println(" -----------------------------------------------");
    }

    /**
     * This function runs twice and adding 2 data sets in 1 Array.
     * @param line 64 columns in line for example: 0,0,5,14,10,1,0,0,0,2,15,15,15.....
     * @param lineNumber current number of dataSet line for example: 772
     *
     */
    private static void addLineToData(String line, int lineNumber){
        String[] integerString = line.split( "," );
        for (int column = 0; column < integerString.length; column++){
            int dataRow = Integer.parseInt( integerString[column] );
            dataSet[column][lineNumber] = dataRow;
        }
    }

    /**
     * Reading Data Sets with scanner
     */
    private static void ReadDataSets(){

        //Data Sets
        File trainSet = new File("C:\\Users\\Defender\\IdeaProjects\\AI_Coursework_2\\src\\cw2DataSet1.csv"); //set 1
        File testSet = new File("C:\\Users\\Defender\\IdeaProjects\\AI_Coursework_2\\src\\cw2DataSet2.csv"); //set 2

        try{
            int lineNumber = 0;

            //Read Set 1
            Scanner Test = new Scanner(testSet);
            //now go through each line of the results
            while (Test.hasNextLine()) {
                //get the data out of the CSV file so I can access it
                String inputLine = Test.nextLine();
                addLineToData(inputLine,lineNumber);
                lineNumber++;
                //executes each line
            }

            //Read set 2
            Scanner Train = new Scanner(trainSet);
            //now go through each line of the results
            while (Train.hasNextLine()) {
                //get the data out of the CSV file so I can access it
                String inputLine = Train.nextLine();
                addLineToData(inputLine,lineNumber);
                lineNumber++;
                //executes each line
            }

        }
        //Throws Exception If File Not Found
        catch (FileNotFoundException ex) {
            ex.printStackTrace();
        }
    }

    /**
     * Main Function
     */
    public static void main(String[] args) throws Exception {
        // Current time
        long start = System.currentTimeMillis();
        for (int i = 0; i <5; i++) {
            Thread.sleep(60);
        }
        //Execute functions below
        ReadDataSets();
        categorise();
        // finding the time after the functions has executed
        long end = System.currentTimeMillis();
        //finding the time difference and converting it into seconds
        float sec = (end - start) / 1000F; System.out.println("Time taken in execution "+ sec + " seconds");
    }
}
