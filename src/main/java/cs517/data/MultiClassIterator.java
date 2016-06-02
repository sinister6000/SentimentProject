package cs517.data;

import cs517.data.DataSetManager;
import org.apache.commons.io.FileUtils;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.NoSuchElementException;

/**
 * Created by Renita on 5/24/16.
 */

/**
 * Iterator for a DataSetManager
 */
public class MultiClassIterator implements DataSetIterator {
    private DataSetManager dm;
    private final int batchSize;

    private int cursor = 0;


    /**
     * Constructor
     *
     * @param dataSetManager the DataSetManager object that contains all the reviews
     * @param wordVectors    WordVectors object
     * @param batchSize      Size of each minibatch for training
     * @param truncateLength maximum length for a movie review
     * @param train          If true: return the training data. If false: return the testing data.
     */
    public MultiClassIterator(DataSetManager dataSetManager,
                              WordVectors wordVectors,
                              int batchSize,
                              int truncateLength,
                              boolean train)
            throws IOException {

        this.batchSize = batchSize;

        File f1 = new File("src/main/resources/movieData/maasDataset/splits/" + (train ? "train/" : "test/") + "1.txt");
        File f2 = new File("src/main/resources/movieData/maasDataset/splits/" + (train ? "train/" : "test/") + "2.txt");
        File f3 = new File("src/main/resources/movieData/maasDataset/splits/" + (train ? "train/" : "test/") + "3.txt");
        File f4 = new File("src/main/resources/movieData/maasDataset/splits/" + (train ? "train/" : "test/") + "4.txt");
        File f7 = new File("src/main/resources/movieData/maasDataset/splits/" + (train ? "train/" : "test/") + "7.txt");
        File f8 = new File("src/main/resources/movieData/maasDataset/splits/" + (train ? "train/" : "test/") + "8.txt");
        File f9 = new File("src/main/resources/movieData/maasDataset/splits/" + (train ? "train/" : "test/") + "9.txt");
        File f10 = new File("src/main/resources/movieData/maasDataset/splits/" + (train ? "train/" : "test/") + "10.txt");
        positiveFiles = new File[]{f7, f8, f9, f10};
        negativeFiles = new File[]{f1, f2, f3, f4};

        tokenizerFactory = new DefaultTokenizerFactory();
        tokenizerFactory.setTokenPreProcessor(new CommonPreprocessor());
    }


    /**
     * Yields the next batch of DataSet objects. Basically, same as a typical next(), but can request more than one object at a time.
     *
     * @param num the size of the batch to return
     * @return
     */
    @Override
    public DataSet next(int num) {
        if (cursor >= positiveFiles.length + negativeFiles.length) throw new NoSuchElementException();
        try {
            return nextDataSet(num);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * Helper function for next(num).
     *
     * @param num size of batch to return
     * @return batch of DataSet objects
     * @throws IOException
     */
    private DataSet nextDataSet(int num) throws IOException {
        //First: load reviews to String. Alternate positive and negative reviews
        List<String> reviews = new ArrayList<>(num);
        for (int i = 0; i < num && cursor < totalExamples(); i++) {
            if (cursor % 2 == 0) {
                //Load positive review
                int posReviewNumber = cursor / 2;
                String review = FileUtils.readFileToString(positiveFiles[posReviewNumber]);
                reviews.add(review);
            } else {
                //Load negative review
                int negReviewNumber = cursor / 2;
                String review = FileUtils.readFileToString(negativeFiles[negReviewNumber]);
                reviews.add(review);
            }
            cursor++;
        }

        //Second: tokenize reviews and filter out unknown words
        List<List<String>> allTokens = new ArrayList<>(reviews.size());
        int maxLength = 0;
        for (String s : reviews) {
            List<String> tokens = tokenizerFactory.create(s).getTokens();
            List<String> tokensFiltered = new ArrayList<>();
            for (String t : tokens) {
                if (wordVectors.hasWord(t)) tokensFiltered.add(t);
            }
            allTokens.add(tokensFiltered);
            maxLength = Math.max(maxLength, tokensFiltered.size());
        }

        //If longest review exceeds 'truncateLength': only take the first 'truncateLength' words
        if (maxLength > truncateLength) maxLength = truncateLength;

        //Create data for training
        //Here: we have reviews.size() examples of varying lengths
        INDArray features = Nd4j.create(reviews.size(), vectorSize, maxLength);
        INDArray labels = Nd4j.create(reviews.size(), 2, maxLength);    //Two labels: positive or negative
        //Because we are dealing with reviews of different lengths and only one output at the final time step: use padding arrays
        //Mask arrays contain 1 if data is present at that time step for that example, or 0 if data is just padding
        INDArray featuresMask = Nd4j.zeros(reviews.size(), maxLength);
        INDArray labelsMask = Nd4j.zeros(reviews.size(), maxLength);

        int[] temp = new int[2];
        for (int i = 0; i < reviews.size(); i++) {
            List<String> tokens = allTokens.get(i);
            temp[0] = i;
            //Get word vectors for each word in review, and put them in the training data
            for (int j = 0; j < tokens.size() && j < maxLength; j++) {
                String token = tokens.get(j);
                INDArray vector = wordVectors.getWordVectorMatrix(token);
                features.put(new INDArrayIndex[]{NDArrayIndex.point(i), NDArrayIndex.all(), NDArrayIndex.point(j)}, vector);

                temp[1] = j;
                featuresMask.putScalar(temp, 1.0);  //Word is present (not padding) for this example + time step -> 1.0 in features mask
            }

            int idx = (positive[i] ? 0 : 1);
            int lastIdx = Math.min(tokens.size(), maxLength);
            labels.putScalar(new int[]{i, idx, lastIdx - 1}, 1.0);   //Set label: [0,1] for negative, [1,0] for positive
            labelsMask.putScalar(new int[]{i, lastIdx - 1}, 1.0);   //Specify that an output exists at the final time step for this example
        }

        return new DataSet(features, labels, featuresMask, labelsMask);
    }

    @Override
    public int totalExamples() {
        return positiveFiles.length + negativeFiles.length;
    }

    @Override
    public int inputColumns() {
        return vectorSize;
    }

    @Override
    public int totalOutcomes() {
        return 2;
    }

    @Override
    public void reset() {
        cursor = 0;
    }

    @Override
    public int batch() {
        return batchSize;
    }

    @Override
    public int cursor() {
        return cursor;
    }

    @Override
    public int numExamples() {
        return totalExamples();
    }

    @Override
    public void setPreProcessor(DataSetPreProcessor preProcessor) {
        throw new UnsupportedOperationException();
    }

    @Override
    public List<String> getLabels() {
        return Arrays.asList("positive", "negative");
    }

    @Override
    public boolean hasNext() {
        return cursor < numExamples();
    }

    @Override
    public DataSet next() {
        return next(batchSize);
    }

    @Override
    public void remove() {

    }

    /**
     * Convenience method for loading review to String
     */
    public String loadReviewToString(int index) throws IOException {
        File f;
        if (index % 2 == 0) f = positiveFiles[index / 2];
        else f = negativeFiles[index / 2];
        return FileUtils.readFileToString(f);
    }

    /**
     * Convenience method to get label for review
     */
    public boolean isPositiveReview(int index) {
        return index % 2 == 0;
    }

}
