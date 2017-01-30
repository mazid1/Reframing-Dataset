package reframing;

import java.io.FileReader;
import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.util.Vector;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.core.OptionHandler;
import weka.core.Utils;
import weka.filters.Filter;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.trees.J48;

/**
 *
 * @author Mazid
 */
public class Main {

    public static void main(String[] args) throws IOException, ClassNotFoundException, Exception {
        
        DataPreprocessor dp = new DataPreprocessor();
        dp.preprocess("data/chronic_kidney_disease.arff", "data/edited.arff");
        dp.age_0_to_30("data/edited.arff", "data/age_0_to_30.arff");
        dp.age_greater_than_70("data/edited.arff", "data/age_more_than_70.arff");
        dp.createModel("data/age_0_to_30.arff", "data/age_0_to_30.model");
        
        // load train data set
        Instances train = new Instances(
                            new BufferedReader(
                                new FileReader("data/age_0_to_30.arff")));
        train.setClassIndex(train.numAttributes() - 1);
        
        // load test data set
        Instances test = new Instances(
                            new BufferedReader(
                                new FileReader("data/age_more_than_70.arff")));
        test.setClassIndex(test.numAttributes() - 1);
        
        // deserialize model
        ObjectInputStream ois = new ObjectInputStream(
                           new FileInputStream("data/age_0_to_30.model"));
        Classifier cls = (Classifier) ois.readObject();
        ois.close();
        
        // evaluate classifier and print some statistics
        Evaluation eval = new Evaluation(train);
        eval.evaluateModel(cls, test);
        System.out.println(eval.toSummaryString("\nResults\n======\n", false));
    }
    
}
