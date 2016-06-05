package cs517.data;

import edu.stanford.nlp.ling.CoreAnnotations.SentencesAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TextAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TokensAnnotation;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.util.CoreMap;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.accum.Sum;
import org.nd4j.linalg.factory.Nd4j;

import java.util.List;
import java.util.Properties;
import java.util.Scanner;
import java.util.regex.MatchResult;
import java.util.regex.Pattern;

/**
 * Created by allen on 5/18/2016.
 */

/**
 * Class to store a review. Unlabeled reviews get a polarity of -1. During the creation of a Review object,
 * we tokenize and translate to word2vec vectors.
 */

public class Review {

    String id;
    int score;
    int polarity;
    String reviewText;
    INDArray reviewVecs;


    public Review() {
    }

    /**
     * Constructs a Review from a tab delimited String.
     * @param lineFromMaas a review from the Maas dataset (1 line per review)
     */
    public Review(String lineFromMaas) {
        Scanner sc = new Scanner(lineFromMaas);
        Pattern p = Pattern.compile("\"(\\d+_(\\d+))\"\\t([01])?\\t?\"(.*)\"$");
        sc.findInLine(p);
        MatchResult result = sc.match();
        id = result.group(1);
        score = Integer.parseInt(result.group(2));
        reviewText = result.group(4);
        try {
            polarity = Integer.parseInt(result.group(3));
        } catch (NumberFormatException e) {
            polarity = -1;
        } finally {
            sc.close();
        }
    }

    /**
     * Converts reviewText to vector form. This represents a review as a time
     * series of sentences for input into a GravesLSTM layer.
     *
     * @param vsm vector space model used to vectorize the Review
     * @param maxSentences
     */
    void vectorizeReview(WordVectors vsm, int maxSentences) {
        System.out.println("vectorizing review " + id);

        int vectorSize = vsm.lookupTable().layerSize();
        System.out.println("vectorSize = " + vectorSize);

        // define pipeline properties, then create pipeline
        Properties props = new Properties();
        props.put("annotators", "tokenize, ssplit");
        StanfordCoreNLP pipeline = new StanfordCoreNLP(props);

        // create Annotation
        Annotation annotatedReview = new Annotation(reviewText);

        // NLP pipeline uses Annotation to process the reviewText with the annotators listed above
        pipeline.annotate(annotatedReview);

        /**
         * Each annotator is like a layer of extra information on top of the original text.
         *    -The 'tokenize' annotator marks word boundaries.
         *    -The 'ssplit' annotator marks sentence boundaries.
         *    -The 'pos' annotator superimposes part of speech info on top of the text.
         *    -and so on...
         *
         * So, the original text remains unchanged, and we use Annotations to see "enhanced" versions of the text.
         * You can also layer annotators on top of each other (e.g. view sentences of pos-tagged tokens).
         * Some annotation layers depend on others (ssplit needs the information from tokenize in order to do its thing).
         */

        List<CoreMap> sentences = annotatedReview.get(SentencesAnnotation.class);
        int sentenceCursor = 0;

        // create NDArray to store sentence vectors. This will serve as the Review's
        // vector representation.
        INDArray sentenceVecs = Nd4j.create(sentences.size(), vectorSize);

        /**
         * For each sentence, calculate a representative vector
         * (here, we'll just average the vectors of the sentence's tokens).
         */
        for (CoreMap s : sentences) {
//            System.out.println("  " + s);

            // create NDArray to store token vectors. Later, we can take the average along dimension 1
            // in order to create a vector for the sentence.
            INDArray tokenVecs = Nd4j.create(s.get(TokensAnnotation.class).size(), vectorSize);
//            System.out.println("tokenVecs shape: " + tokenVecs.shape()[0] + " x " + tokenVecs.shape()[1]);
            int cursor = 0;
            for (CoreLabel token : s.get(TokensAnnotation.class)) {
                String tokenText = token.getString(TextAnnotation.class).toLowerCase();
                if (vsm.hasWord(tokenText)) {
                    INDArray rowVector = vsm.getWordVectorMatrix(tokenText);
//                    System.out.println("shape " + rowVector.shape()[0] + " x " + rowVector.shape()[1]);
                    tokenVecs.putRow(cursor++, rowVector);
                }
            }

            // get the average of each column
            INDArray sumOfColumns = Nd4j.getExecutioner().exec(new Sum(tokenVecs), 0);
            INDArray senVec = sumOfColumns.divi(cursor);
            sentenceVecs.putRow(sentenceCursor++, senVec);
            if (sentenceCursor >= maxSentences) {
                break;
            }
        }

        reviewVecs = sentenceVecs;
        System.out.println(reviewVecs);
        System.out.println(id + " DONE!\n");
    }
}
