package cs517;

import cs517.data.DataSetManager;

import java.io.File;
import java.io.IOException;

/**
 * Created by allen on 6/4/2016.
 */
public class MainProgram {


    public static void main(String[] args) {
//
//        int maxLength = 50;
//        int vectorSize = 100;
//        int batchSize = 32;
//
//
//
        DataSetManager dm = new DataSetManager();
        File ldf = new File("src/main/resources/movieData/toyMaasDataset/toyLabeledTrainData.tsv");
        File tdf = new File("src/main/resources/movieData/toyMaasDataset/toyTestData.tsv");
        try {
            dm.importData(ldf);
            dm.importData(tdf);
        } catch (IOException e) {
            e.printStackTrace();
        }


//
//        List<DataSetIterator> iterators = dm.makeIterators(batchSize);
//        DataSetIterator train = iterators.get(0);
//        DataSetIterator cv = iterators.get(1);
//        DataSetIterator test = iterators.get(2);
//
//        RNN myNN = new RNN();
//        myNN.net.init();
//        myNN.vectorSize = vectorSize;
//
////            WordVectors wordVectors = WordVectorSerializer.loadTxtVectors(new File("sentimentWordVectors.txt"));
////            vectorSize = wordVectors.lookupTable().layerSize();
////        myNN.net.init();
////            myNN.net.setListeners(new ScoreIterationListener(1));
////            myNN.net.setListeners(new HistogramIterationListener(1));
//
//        System.out.println("Starting training");
//        for (int i = 0; i < myNN.nEpochs; i++) {
//            myNN.net.fit(train);
//            train.reset();
//            System.out.println("Epoch " + i + " complete. Starting evaluation:");
//
//            //Run evaluation. This is on 25k reviews, so can take some time
//            Evaluation evaluation = new Evaluation();
//
//            while (test.hasNext()) {
//
//                DataSet t = test.next();
//                INDArray features = t.getFeatureMatrix();
//                INDArray labels = t.getLabels();
//                INDArray inMask = t.getFeaturesMaskArray();
//                INDArray outMask = t.getLabelsMaskArray();
//                INDArray predicted = myNN.net.output(features, false, inMask, outMask);
//
//                evaluation.evalTimeSeries(labels, predicted, outMask);
//                i++;
//            }
//            test.reset();
//
//            System.out.println(evaluation.stats());
//        }
//        System.out.println("----- Example complete -----");

    }


}

