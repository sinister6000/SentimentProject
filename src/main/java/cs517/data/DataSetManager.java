package cs517.data;

import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;

import java.io.*;
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
    List<String> revIDs;

    public DataSetManager() {
        reviews = new HashMap<>();
        ratingBins = new HashMap<>();
        revIDs = new ArrayList<>();
    }

    /**
     * Imports reviews from a file.
     *
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

        } catch (FileNotFoundException e) {
            System.err.println("importData could not find file");
            e.printStackTrace();
        }

        // quick check to see if import went ok
        if (reviews.size() != revIDs.size()) {
            System.out.println("Warning: Size mismatch between reviews and revIDs!!!!");
        }
    }


    /**
     * For all reviews in DataSetManager, convert the review text to its wordVector representation.
     * Basically, we break the review up into sentences. Each sentence is made up of tokens. So, we look
     * up a word vector for each token. Then, we can represent a sentence as the centroid of its words'
     * vectors. Finally, the review is represented as a chain of sentences.
     *
     * Note that all sentences (regardless of length) are represented by a 300-dimensional vector.
     * So a review is a (S x 300) dimensional INDArray, where S == # of sentences in the review.
     *
     * Store this INDArray in the Review object.
     *
     * @param vsm Vector Space Model that has all the wordVectors.
     */
    public void reviews2wordVectors(WordVectors vsm, int maxLength) {
        StanfordCoreNLP pipeline = new StanfordCoreNLP(props);
        // for each review in reviews
            // pass through Stanford CoreNLP, giving us sentences and tokens

            // for each sentence (up to maxLength, which is max # of sentences),
                // for each token
                    // lookup word vector, save to temp INDArray, tokenVectors

                // sentence vector = average of the tokenVectors in the temp INDArray
                // store sentence vector on another INDArray, sentenceVectors
            // after all sentences processed, sentenceVectors represents the whole review
            // store sentenceVectors in Review object as a field


        // at this point, all Review objects in our DataSetManager have been updated to include their
        // vector representations
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


    public static void makeTrainingSplits() {
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

    public static void makeTestingSplits() {
        DataSetManager testingDM = new DataSetManager();
        File ldf = new File("src/main/resources/movieData/maasDataset/testData.tsv");
        testingDM.importData(ldf);
        String pathForSplitData = "src/main/resources/movieData/maasDataset/splits/";
        File dir = new File(pathForSplitData);
        dir.mkdirs();
        testingDM.toMultiClassFiles(pathForSplitData, false);
    }

    public static void main(String[] args) {
//        makeTrainingSplits();
//        makeTestingSplits();
        DataSetManager trainingDM = new DataSetManager();
        File ldf = new File("src/main/resources/movieData/maasDataset/labeledTrainData.tsv");
        trainingDM.importData(ldf);
        File tdf = new File("src/main/resources/movieData/maasDataset/testData.tsv");
        trainingDM.importData(tdf);

        final String WORD_VECTORS_PATH = "C:/Docs/School/CSUPomona/CS517/NLPProject/data/GoogleNews-vectors-negative300";
        WordVectors googleWordVectors = WordVectorSerializer.loadGoogleModel(new File(WORD_VECTORS_PATH), false);


        int MAX_REVIEW_LENGTH = 100;  // 100 sentences
        trainingDM.reviews2wordVectors(googleWordVectors, MAX_REVIEW_LENGTH);
    }


}
