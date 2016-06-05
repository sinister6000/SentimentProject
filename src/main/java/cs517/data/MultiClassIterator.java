package cs517.data;

import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.NoSuchElementException;

//import java.io.File;


/**
 * This is a DataSetIterator that is specialized for the Stanford Maas IMDB review dataset.
 * It takes a DataSetManager and a list of Review IDs and generates training data sets for a neural network.
 * The training sets are structured as follows:
 *      Inputs/features: variable-length time series of vectors. Each vector represents a sentence
 *                       in the review.
 *      Labels/target: a single classification (1, 2, 3, 4, 7, 8, 9, or 10) represented as a
 *                     one-hot set. This is the desired output of the neural network.
 */
public class MultiClassIterator implements DataSetIterator {
    private DataSetManager dm;
    private int batchSize;
    private List<String> reviewsToIterate;
    private int cursor = 0;
    private int vectorSize = 300;
    private int maxLength = 50;


    /**
     * Constructor
     *
     * @param dataSetManager    contains the actual Review objects needed to create data sets
     * @param fromIndex         low endpoint (inclusive) of the subList of revIDs
     * @param toIndex           high endpoint (exclusive) of the subList of revIDs
     * @param batchSize         mini-batch size
     * @throws IOException
     */
    public MultiClassIterator(DataSetManager dataSetManager, int fromIndex, int toIndex, int batchSize) {
        dm = dataSetManager;
        this.batchSize = batchSize;
        reviewsToIterate = dm.shuffledRevIDs.subList(fromIndex, toIndex);
    }


    /**
     * Yields a mini-batch Data Set.
     *
     * @param num the number of reviews being returned in this Data Set
     * @return
     */
    @Override
    public DataSet next(int num) {
        if (cursor >= reviewsToIterate.size()) {
            throw new NoSuchElementException();
        }
        return nextDataSet(num);
    }

    /**
     * Helper function for above.
     *
     * @param num == batchSize
     * @return
     */
    private DataSet nextDataSet(int num) {

        // create data for training
        INDArray features = Nd4j.create(num, vectorSize, maxLength);
        INDArray labels = Nd4j.create(num, 8, maxLength);
        

         /*
         Need to pad features and labels arrays with masks because the network is expecting a time series input of a certain
         time length. Reviews vary in the # of sentences they contain, so we pad short ones with 0's.
         Also, we pad the output to time it so that it arrives simultaneously with the end of the input series.
         Mask arrays contain 1 if data is present at that time step for that example, or 0 if data is just padding
         */
        INDArray featuresMask = Nd4j.zeros(num, maxLength);
        INDArray labelsMask = Nd4j.zeros(num, maxLength);

        for (int i = 0; i < batchSize; ++i) {
            Review rev = dm.reviews.get(reviewsToIterate.get(i));
            INDArray revVectors = rev.reviewVecs;
            features.put(new INDArrayIndex[]{
                    NDArrayIndex.point(i),
                    NDArrayIndex.all(),
                    NDArrayIndex.interval(0, revVectors.shape()[1])}, revVectors);
            featuresMask.put(new INDArrayIndex[]{NDArrayIndex.interval(0, revVectors.shape()[1])}, 1.0);

            int revScore = rev.score;
            labels.put(new INDArrayIndex[]{
                    NDArrayIndex.point(i),
                    NDArrayIndex.all(),
                    NDArrayIndex.point(revVectors.shape()[1] - 1)}, oneHot(revScore));
            labelsMask.putScalar(revVectors.shape()[1] - 1, 1.0);
        }


        DataSet result = new DataSet(features, labels, featuresMask, labelsMask);
        return result;
    }

    /**
     * Helper method to turn a movie rating score into a one-hot vector.
     *    1 -> [ 1 0 0 0 0 0 0 0 ]
     *    2 -> [ 0 1 0 0 0 0 0 0 ]
     *    3 -> [ 0 0 1 0 0 0 0 0 ]
     *    4 -> [ 0 0 0 1 0 0 0 0 ]
     *    7 -> [ 0 0 0 0 1 0 0 0 ]
     *    8 -> [ 0 0 0 0 0 1 0 0 ]
     *    9 -> [ 0 0 0 0 0 0 1 0 ]
     *   10 -> [ 0 0 0 0 0 0 0 1 ]
     * 
     * @param score
     * @return
     */
    private INDArray oneHot(int score) {
        INDArray label = Nd4j.zeros(8);
        switch (score) {
            case 1:
                label.putScalar(0, 1);
                break;
            case 2:
                label.putScalar(1, 1);
                break;
            case 3:
                label.putScalar(2, 1);
                break;
            case 4:
                label.putScalar(3, 1);
                break;
            case 7:
                label.putScalar(4, 1);
                break;
            case 8:
                label.putScalar(5, 1);
                break;
            case 9:
                label.putScalar(6, 1);
                break;
            case 10:
                label.putScalar(7, 1);
                break;
        }
        return label;
    }

    @Override
    public int totalExamples() {
        return reviewsToIterate.size();
    }

    @Override
    public int inputColumns() {
        return vectorSize;
    }

    @Override
    public int totalOutcomes() {
        return 8;
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
        return Arrays.asList("10 (++++)", "9 (+++)", "8 (++)", "7 (+)", "4 (-)", "3 (--)", "2 (---)", "1 (----)");
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
//    public String loadReviewToString(int index) throws IOException {
//        File f;
//        if (index % 2 == 0) f = positiveFiles[index / 2];
//        else f = negativeFiles[index / 2];
//        return FileUtils.readFileToString(f);
//    }
//
//    /**
//     * Convenience method to get label for review
//     */
//    public boolean isPositiveReview(int index) {
//        return index % 2 == 0;
//    }

}
