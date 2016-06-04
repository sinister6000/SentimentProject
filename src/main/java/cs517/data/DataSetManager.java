package cs517.data;

import org.apache.uima.resource.ResourceInitializationException;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
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
            currentReview.vectorizeReview(vsm, maxSentences);

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
        int cvStart = trainEnd + 1;
        int cvEnd = (int) (0.8 * revCount);
        int testStart = cvEnd + 1;
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
        /**
         * TODO: adapt MultiClassIterator into this form
         */


        return new MultiClassIterator(this, fromIndex, toIndex, batchSize);
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

//    public void reviews2wordVectors(int maxLength) throws ResourceInitializationException {

        /**
         * TODO: I think i see where your confusion came from. It's my bad, I was unclear about how
         * and why I was changing the logic and flow of their example. And I chose variable names that
         * added to confusion...
         *
         * Since their example was a main() that was meant to just be run once,
         * they view the creation of a vector representation of a review
         * as a part of iterating through the training corpus. While this is fine for running it once, I was
         * anticipating having to try different NN configs, etc. So, I was thinking about creating the vector
         * representations for the reviews earlier in the whole process and saving these vector reps inside
         * the Review objects themselves. This is mainly to save time on recomputing these vectors every time
         * we try a different NN. But also, conceptually, their SentimentExampleIterator is doing too much.
         *
         * To further confuse us, in SentimentExampleIterator, when they say "reviews" they are referring
         * to a temporary mini-batch of reviews that they are preparing to send as input to the RNN.
         * On top of all this, they are also creating the desired output vector (they call it temp[]).
         * Finally, they combine the mini-batch of vectorized reviews, with a mini-batch of desired outputs
         * in what is unfortunately called a "DataSet." DataSet is ambiguous too because it can refer to a single
         * instance of an INPUT/OUTPUT pair, or a mini-batch of INPUTS/mini-batch of OUTPUTS collection.
         */


        /********************************************
         *
         * So let's move vectorizing over to the Review class.
         * This function will just iterate through all
         * Review objects and tell them to vectorize themselves.
         *
         ********************************************/

        /********************************************************
         * Actually, this is probably better as part of the importData() method.
         * Since we already perform an entire pass through all the reviews in order to import,
         * we can vectorize during that pass-through.
         ******************************************************
         */
//        final String WORD_VECTORS_PATH = "/users/Renita/GoogleNews-vectors-negative300.bin";
//        final String WORD_VECTORS_PATH = "C:/Docs/School/CSUPomona/CS517/NLPProject/data/GoogleNews-vectors-negative300.bin";
//        final String WORD_VECTORS_PATH = "sentimentWordVectors.txt";
//        WordVectors googleWordVectors = null;
//        try {
////            googleWordVectors = WordVectorSerializer.loadGoogleModel(new File(WORD_VECTORS_PATH), true);
//            googleWordVectors = WordVectorSerializer.loadTxtVectors(new File(WORD_VECTORS_PATH));
//        } catch (IOException e) {
//            System.err.println("Can't load Google Word Vectors from file.");
//            e.printStackTrace();
//        }
//        for (Review r : reviews.values()) {
//            r.vectorizeReview(googleWordVectors, maxLength);
//        }
//    } // end reviews2wordVectors()





    /***********************************************************

    TokenizerFactory tokenizerFactory = new DefaultTokenizerFactory();
        tokenizerFactory.setTokenPreProcessor(new CommonPreprocessor());
//
//        tokenizerFactory.setTokenPreProcessor(new TokenPreProcess() {
//            @Override
//            public String preProcess(String token) {
//                token = token.toLowerCase();
//                String base = preProcess(token);
//                base = base.replace("\\d", "d");
//                if(base.endsWith("ly") || base.endsWith("ing")) {
//                    System.out.println("token");
//                }
//                    return base;
//            }
//        });


        List<List<String>> allTokens = new ArrayList<>(reviews.size());

        for (Review r: reviews.values()){
            List<String> tokens = tokenizerFactory.create(r.reviewText).getTokens();
            List<String> tokensFiltered = new ArrayList<>();
            for(String t : tokens ){
                if(vsm.hasWord(t)) tokensFiltered.add(t);
            }
            allTokens.add(tokensFiltered);
        }


        int vectorSize = vsm.lookupTable().layerSize();

        //Here: we have reviews.size() examples of varying lengths
        INDArray featureVector = Nd4j.create(reviews.size(), vectorSize, maxLength);
        INDArray labels = Nd4j.create(reviews.size(), 2, maxLength);    //Two labels: positive or negative

        INDArray featuresMask = Nd4j.zeros(reviews.size(), maxLength);
        INDArray labelsMask = Nd4j.zeros(reviews.size(), maxLength);

        int[] temp = new int[2];
        for( int i=0; i<reviews.size(); i++ ){
            List<String> tokens = allTokens.get(i);
            temp[0] = i;
            //Get word vectors for each word in review, and put them in the training data
            for( int j=0; j<tokens.size() && j<maxLength; j++ ){
                String token = tokens.get(j);
                INDArray vector = vsm.getWordVectorMatrix(token);
                featureVector.put(new INDArrayIndex[]{NDArrayIndex.point(i), NDArrayIndex.all(), NDArrayIndex.point(j)}, vector);

                temp[1] = j;
                featuresMask.putScalar(temp, 1.0);  //Word is present (not padding) for this example + time step -> 1.0 in features mask
            }
        }

//        for (Review r: reviews.values()) {
//            Tokenizer tokenizer = tokenizerFactory.tokenize(r.getReviewText());
//            List<String> tokens = tokenizer.getTokens();
//
//        }

        //StanfordCoreNLP pipeline = new StanfordCoreNLP(props);
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
****************************************************/

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
