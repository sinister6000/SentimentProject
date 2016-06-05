package cs517.data;

import org.apache.uima.resource.ResourceInitializationException;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.*;

/**
 * Created by allen on 5/17/2016.
 */



public class DataSetManager {


    Map<String, Review> reviews;

    /**
     * ratingBins maps a review score (1, 2, 3, 4, 7, 8, 9, 10) to a list of review IDs.
     * This Map will be used for most of the processing, then we use the IDs to reference the other Map
     * to get the whole review.
     */
    Map<Integer, Set<String>> ratingBins;

    /**
     * revIDs is just a list of review IDs. This will be used to create a random ordering of the reviews
     * during the creation of Iterators to feed the neural net.
     */
    Set<String> revIDs;

    List<String> shuffledRevIDs;


    public DataSetManager() {
        reviews = new HashMap<>();
        ratingBins = new HashMap<>();
        revIDs = new HashSet<>();
        shuffledRevIDs = new ArrayList<>();
    }

    /**
     * Imports reviews from a file.
     *
     * @param f File of reviews
     */
    public void importData(File f) throws IOException {
//        final String WORD_VECTORS_PATH = "C:/Docs/School/CSUPomona/CS517/NLPProject/data/GoogleNews-vectors-negative300.bin";
//        WordVectors vsm = WordVectorSerializer.loadGoogleModel(new File(WORD_VECTORS_PATH), true, false);
        final String WORD_VECTORS_PATH = "sentimentWordVectors.txt";
        WordVectors vsm = WordVectorSerializer.loadTxtVectors(new File(WORD_VECTORS_PATH));

        Scanner sc = new Scanner(f);
        sc.useDelimiter(System.getProperty("line.separator"));
        String currentLine = sc.next();   // skip over 1st header line
        Review currentReview;
        while (sc.hasNext()) {
            currentLine = sc.next();
            currentReview = new Review(currentLine);
            currentReview.vectorizeReview(vsm, 75);

            // store Review in map
            reviews.put(currentReview.id, currentReview);

            // add currentReview to proper bin, else create a bin.
            Integer currentScore = currentReview.score;
            if (ratingBins.containsKey(currentScore)) {
                Set<String> temp = ratingBins.get(currentScore);
                temp.add(currentReview.id);
                ratingBins.put(currentScore, temp);
            } else {
                Set<String> temp = new HashSet<>();
                temp.add(currentReview.id);
                ratingBins.put(currentScore, temp);
            }
            // add ID to list
            revIDs.add(currentReview.id);
        }
        // quick check to see if import went ok
        if (reviews.size() != revIDs.size()) {
            System.out.println("Warning: Size mismatch between reviews and revIDs!!!!");
        }

    }

    /**
     * Provides a group of three iterators, which when combined, include all the reviews.
     * First, random shuffles the reviews, then assigns subarrays of the reviews to the
     * training set, cross validation set, and testing set in the following proportions:
     *      60% - training set
     *      20% - cross validation set
     *      20% - testing set
     *
     * @param batchSize size of mini-batch for GravesLSTM network layer.
     * @return [training iter, cv iter, testing iter]
     */
    public List<DataSetIterator> makeIterators(int batchSize) {
        shuffledRevIDs = new ArrayList<>(revIDs);
        Collections.shuffle(shuffledRevIDs);

        int revCount = revIDs.size();
        int trainEnd = (int) (0.6 * revCount);
        int cvStart = trainEnd;
        int cvEnd = (int) (0.8 * revCount);
        int testStart = cvEnd;
        int testEnd = revCount;

        DataSetIterator trainIter = makeDataSetIterator(0, trainEnd, batchSize);
        DataSetIterator cvIter = makeDataSetIterator(cvStart, cvEnd, batchSize);
        DataSetIterator testIter = makeDataSetIterator(testStart, testEnd, batchSize);

        List<DataSetIterator> myIterators = new ArrayList<>();
        myIterators.add(trainIter);
        myIterators.add(cvIter);
        myIterators.add(testIter);

        return myIterators;
    }


    /**
     * Creates an iterator from a subList of shuffledRevIDs.
     *
     * @param fromIndex         low endpoint (inclusive) of the subList
     * @param toIndex           high endpoint (exclusive) of the subList
     * @param batchSize         mini-batch size
     * @return DataSetIterator  iterator that will yield the reviews whose IDs are contained
     *                          within the subList.
     */
    private DataSetIterator makeDataSetIterator(int fromIndex, int toIndex, int batchSize) {
        return new MultiClassIterator(this, fromIndex, toIndex, batchSize);
    }


    /**
     * Outputs reviews to dir, 1 review per file. Intended to be used for polarity.
     * Usage example: to output all positive reviews, first create the directory for the files.
     * <p>
     * DataSetManager allReviews = new DataSetManager();
     * File dataFile = new File('PATH_TO_DATAFILE');
     * allReviews.importData(dataFile);
     * File directory = new File(dataFile.getParent() + "labeled/pos");
     * directory.mkdirs();
     * allReviews.toPolarityFiles(directory, 1);
     *
     * @param dir       directory to write files
     * @param scoreType unlabeled (-1), negative (0), or positive (1)
     */
    public void toPolarityFiles(String dir, int scoreType) {
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


    /**
     * Outputs all reviews to a single file, one review per line.
     *
     * @param fname
     */
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


    /**
     * Output a file for each bin of reviews.
     *
     * @param dir
     * @param train indicates training or testing
     */
    public void toMultiClassFiles(String dir, boolean train) {
        File newDir = new File(dir + (train ? "train" : "test"));
        newDir.mkdirs();
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

            BufferedWriter[] bwArray = new BufferedWriter[]{b1, b2, b3, b4, b7, b8, b9, b10};

            Review currRev;
            for (String revID : reviews.keySet()) {
                currRev = reviews.get(revID);
                int sentimentScore = currRev.score;
                switch (sentimentScore) {
                    case 0:
                        //b0.write(currRev.reviewText + "\n");
                        break;
                    case 1:
                        b1.write(currRev.id + "\t" + currRev.reviewText + "\n");
                        break;
                    case 2:
                        b2.write(currRev.id + "\t" + currRev.reviewText + "\n");
                        break;
                    case 3:
                        b3.write(currRev.id + "\t" + currRev.reviewText + "\n");
                        break;
                    case 4:
                        b4.write(currRev.id + "\t" + currRev.reviewText + "\n");
                        break;
                    case 7:
                        b7.write(currRev.id + "\t" + currRev.reviewText + "\n");
                        break;
                    case 8:
                        b8.write(currRev.id + "\t" + currRev.reviewText + "\n");
                        break;
                    case 9:
                        b9.write(currRev.id + "\t" + currRev.reviewText + "\n");
                        break;
                    case 10:
                        b10.write(currRev.id + "\t" + currRev.reviewText + "\n");
                        break;
                }
            }
            for (BufferedWriter bw : bwArray) {
                bw.close();
            }
        } catch (IOException e) {
            System.err.println("Problem writing Multi-class files.");
            e.printStackTrace();
        }
    }


    public static void makeTrainingSplits() throws IOException {
        DataSetManager trainingDM = new DataSetManager();
        File ldf = new File("src/main/resources/movieData/maasDataset/labeledTrainData.tsv");
        trainingDM.importData(ldf);
        //File udf = new File("src/main/resources/movieData/maasDataset/unlabeledTrainData.tsv");
        //trainingDM.importData(udf);
        String pathForSplitData = "src/main/resources/movieData/maasDataset/splits/";
        File dir = new File(pathForSplitData);
        dir.mkdirs();
        trainingDM.toMultiClassFiles(pathForSplitData, true);
    }

    public static void makeTestingSplits() throws IOException{
        DataSetManager testingDM = new DataSetManager();
        File ldf = new File("src/main/resources/movieData/maasDataset/testData.tsv");
        testingDM.importData(ldf);
        String pathForSplitData = "src/main/resources/movieData/maasDataset/splits/";
        File dir = new File(pathForSplitData);
        dir.mkdirs();
        testingDM.toMultiClassFiles(pathForSplitData, false);
    }

    public static void main(String[] args) throws IOException, ResourceInitializationException {
//        makeTrainingSplits();
//        makeTestingSplits();
        DataSetManager trainingDM = new DataSetManager();
        File ldf = new File("src/main/resources/movieData/toyMaasDataset/toyLabeledTrainData.tsv");
        trainingDM.importData(ldf);
        File tdf = new File("src/main/resources/movieData/toyMaasDataset/toyTestData.tsv");
        trainingDM.importData(tdf);

//        final String WORD_VECTORS_PATH = "/users/Renita/GoogleNews-vectors-negative300.bin";
//        final String WORD_VECTORS_PATH = "C:/Docs/School/CSUPomona/CS517/NLPProject/data/GoogleNews-vectors-negative300";
//        WordVectors googleWordVectors = WordVectorSerializer.loadGoogleModel(new File(WORD_VECTORS_PATH), false);


//        int MAX_REVIEW_LENGTH = 100;  // 100 sentences
//        trainingDM.reviews2wordVectors(MAX_REVIEW_LENGTH);
    }

}
