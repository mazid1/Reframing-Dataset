package reframing;

import java.io.FileReader;
import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.ObjectInputStream;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Instances;


/**
 *
 * @author Mazid
 */
public class Main {

    public static void main(String[] args) throws IOException, ClassNotFoundException, Exception {
        
        /////////////////////////////////////
        // Chronic kindey disease data
        /////////////////////////////////////
        
        DataPreprocessor dp = new DataPreprocessor();
        // remove some unecessary attributes and instaces with missing values
        dp.preprocess("data/chronic_kidney_disease.arff", "data/edited.arff");
        // generate new arff file for "age" value from 0 to 30
        dp.age_0_to_30("data/edited.arff", "data/age_0_to_30.arff");
        // generate new arff file for "age" value greater than 70
        dp.age_greater_than_70("data/edited.arff", "data/age_more_than_70.arff");
        // create new model using train data
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
        System.out.println("Chronic kidney disease:");
        System.out.println(eval.toSummaryString("\nResults\n======\n", false));
        
        Reframing rf = new Reframing();
        rf.reframing(train, test, cls);
        
        ReframingQuadratic rq = new ReframingQuadratic();
        rq.reframing(train, test, cls);
        
        /////////////////////////////////////////
        // Heart disease data
        /////////////////////////////////////////
        
        // load train data set
        train = new Instances(
                            new BufferedReader(
                                new FileReader("data/processed.cleveland.arff")));
        train.setClassIndex(train.numAttributes() - 1);
        
        // create model
        dp = new DataPreprocessor();
        dp.createModel("data/processed.cleveland.arff", "data/cleveland.model");
        
        // load test data set
        test = new Instances(
                            new BufferedReader(
                                new FileReader("data/processed.hungarian.arff")));
        test.setClassIndex(test.numAttributes() - 1);
        
        // deserialize model
        ois = new ObjectInputStream(
                           new FileInputStream("data/cleveland.model"));
        cls = (Classifier) ois.readObject();
        ois.close();
        
        // evaluate classifier and print some statistics
        eval = new Evaluation(train);
        eval.evaluateModel(cls, test);
        System.out.println("Heart disease: (Hungarian)");
        System.out.println(eval.toSummaryString("\nResults\n======\n", false));
        
        rf = new Reframing();
        rf.reframing(train, test, cls);
        
        rq = new ReframingQuadratic();
        rq.reframing(train, test, cls);
        
        
        /////////////////////////////////////////
        // Synthetic data
        /////////////////////////////////////////
        
        // load train data set
        train = new Instances(
                            new BufferedReader(
                                new FileReader("data/synthetic_train.arff")));
        train.setClassIndex(train.numAttributes() - 1);
        
        // create model
        dp = new DataPreprocessor();
        dp.createModel("data/synthetic_train.arff", "data/synthetic.model");
        
        // load test data set
        test = new Instances(
                            new BufferedReader(
                                new FileReader("data/synthetic_test.arff")));
        test.setClassIndex(test.numAttributes() - 1);
        
        // deserialize model
        ois = new ObjectInputStream(
                           new FileInputStream("data/synthetic.model"));
        cls = (Classifier) ois.readObject();
        ois.close();
        
        // evaluate classifier and print some statistics
        eval = new Evaluation(train);
        eval.evaluateModel(cls, test);
        System.out.println("Synthetic data");
        System.out.println(eval.toSummaryString("\nResults\n======\n", false));
        
        rf = new Reframing();
        rf.reframing(train, test, cls);
        
        rq = new ReframingQuadratic();
        rq.reframing(train, test, cls);
    }
    
}
