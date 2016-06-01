package cs517.data;

import java.io.*;
import java.util.*;

/**
 * Created by allen on 5/17/2016.
 */
public class DatasetManager {
    private Map<String, Review> reviews;

    public DatasetManager() {
        reviews = new HashMap<>();
    }

    /**
     * Imports reviews from a file.
     * @param f File of reviews
     */
    public void importData(File f) {
        try {
            Scanner sc = new Scanner(f);
            sc.useDelimiter(System.getProperty("line.separator"));
            String currentLine = sc.next();   // skip over 1st header line
            Review currentReview;
            while (sc.hasNext()) {
                currentLine = sc.next();
                currentReview = new Review(currentLine);
                reviews.put(currentReview.id, currentReview);
            }
        } catch (FileNotFoundException e) {
            System.err.println("importData could not find file");
            e.printStackTrace();
        }
    }


    /**
     * Outputs reviews to dir, 1 review per file.
     * Usage example: to output all positive reviews, first create the directory for the files.
     *
     *       DatasetManager allReviews = new DatasetManager();
     *       File dataFile = new File('PATH_TO_DATAFILE');
     *       allReviews.importData(dataFile);
     *       File directory = new File(dataFile.getParent() + "labeled/pos");
     *       directory.mkdirs();
     *       allReviews.toSeparateFiles(directory, 1);
     *
     * @param dir directory to write files
     * @param scoreType unlabeled (-1), negative (0), or positive (1)
     */
    public void toSeparateFiles(String dir, int scoreType) {
        if (!reviews.isEmpty()) {
            for (String key : reviews.keySet()) {
                Review currentReview = reviews.get(key);
                if (currentReview.score == scoreType) {
                    File f = new File(dir + currentReview.id + ".txt");
                    try (BufferedWriter bw = new BufferedWriter(new FileWriter(f))) {
                        bw.write(currentReview.reviewText);
                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                }
            }
        }
    }

    public void toSingleFile(String fname) {
        if (!reviews.isEmpty()) {
            File f = new File(fname);
            try (BufferedWriter bw = new BufferedWriter(new FileWriter(f))) {
                for (String key : reviews.keySet()) {
                    Review currentReview = reviews.get(key);
                    bw.write(currentReview.reviewText);
                    bw.write("\n");
                }
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }


    public void toGroupFiles(String dir, boolean train) {
        File newDir = new File(dir + (train ? "train" : "test"));
        newDir.mkdirs();
        List<BufferedWriter> bwList = new ArrayList<>();
        try {
            //BufferedWriter b0 = new BufferedWriter(new FileWriter(new File(dir + (train ? "train" : "test") + "/0.txt")));
            BufferedWriter b1 = new BufferedWriter(new FileWriter(newDir + "/1.txt"));
            BufferedWriter b2 = new BufferedWriter(new FileWriter(newDir + "/2.txt"));
            BufferedWriter b3 = new BufferedWriter(new FileWriter(newDir + "/3.txt"));
            BufferedWriter b4 = new BufferedWriter(new FileWriter(newDir + "/4.txt"));
            BufferedWriter b7 = new BufferedWriter(new FileWriter(newDir + "/7.txt"));
            BufferedWriter b8 = new BufferedWriter(new FileWriter(newDir + "/8.txt"));
            BufferedWriter b9 = new BufferedWriter(new FileWriter(newDir + "/9.txt"));
            BufferedWriter b10 = new BufferedWriter(new FileWriter(newDir + "/10.txt"));

            //bwList.add(b0);
            bwList.add(b1);
            bwList.add(b2);
            bwList.add(b3);
            bwList.add(b4);
            bwList.add(b7);
            bwList.add(b8);
            bwList.add(b9);
            bwList.add(b10);


            Review currRev;
            for (String revID : reviews.keySet()) {
                currRev = reviews.get(revID);
                int sentimentScore = currRev.score;
                switch (sentimentScore) {
                    case 0:
                        //b0.write(currRev.reviewText + "\n");
                        break;
                    case 1:
                        b1.write(currRev.reviewText + "\n");
                        break;
                    case 2:
                        b2.write(currRev.reviewText + "\n");
                        break;
                    case 3:
                        b3.write(currRev.reviewText + "\n");
                        break;
                    case 4:
                        b4.write(currRev.reviewText + "\n");
                        break;
                    case 7:
                        b7.write(currRev.reviewText + "\n");
                        break;
                    case 8:
                        b8.write(currRev.reviewText + "\n");
                        break;
                    case 9:
                        b9.write(currRev.reviewText + "\n");
                        break;
                    case 10:
                        b10.write(currRev.reviewText + "\n");
                        break;
                }
            }

            for (BufferedWriter bw : bwList) {
                bw.close();
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }


    public static void makeTrainingSplits() {
        DatasetManager dm = new DatasetManager();
        File ldf = new File("src/main/resources/movieData/maasDataset/labeledTrainData.tsv");
        dm.importData(ldf);
        //File udf = new File("src/main/resources/movieData/maasDataset/unlabeledTrainData.tsv");
        //dm.importData(udf);
        String pathForSplitData = "src/main/resources/movieData/maasDataset/splits/";
        File dir = new File(pathForSplitData);
        dir.mkdirs();
        dm.toGroupFiles(pathForSplitData, true);
    }

    public static void makeTestingSplits() {
        DatasetManager dm = new DatasetManager();
        File ldf = new File("src/main/resources/movieData/maasDataset/testData.tsv");
        dm.importData(ldf);
        String pathForSplitData = "src/main/resources/movieData/maasDataset/splits/";
        File dir = new File(pathForSplitData);
        dir.mkdirs();
        dm.toGroupFiles(pathForSplitData, false);
    }

    public static void main(String[] args) {
        makeTrainingSplits();
        makeTestingSplits();
        //File dir = new File(df.getParent() + "/paravec/");
        //System.out.println("dir path: " + dir.getPath());
        //dir.mkdirs();
        //dm.toSeparateFiles(dir.getPath(), 1);
        //dm.toSeparateFiles(dir.getPath(), 0);
        //dm.toSeparateFiles(dir.getPath(), -1);
        //dm.toSingleFile("src/main/resources/movieData/maasDataset/allReviewText.txt");


    }
}
