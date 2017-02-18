package reframing;

import java.io.FileReader;
import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
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
        
        Evaluation eval;
        Reframing rf;
        ReframingQuadratic rq;
        DataPreprocessor dp = new DataPreprocessor();
        Instances train;
        Instances test;
        ObjectInputStream ois;
        Classifier cls;
        
    /*  
        // remove some unecessary attributes and instaces with missing values
        dp.preprocess("data/chronic_kidney_disease.arff", "data/edited.arff");
        // generate new arff file for "age" value from 0 to 30
        dp.age_0_to_30("data/edited.arff", "data/age_0_to_30.arff");
        // generate new arff file for "age" value greater than 70
        //dp.age_greater_than_70("data/edited.arff", "data/age_more_than_70.arff");
        // create new model using train data
        dp.createModel("data/age_0_to_30.arff", "data/age_0_to_30.model");
        
        // load train data set
        train = new Instances(
                            new BufferedReader(
                                new FileReader("data/age_0_to_30.arff")));
        train.setClassIndex(train.numAttributes() - 1);
        
        // load test data set
        test = new Instances(
                            new BufferedReader(
                                new FileReader("data/age_more_than_70.arff")));
        test.setClassIndex(test.numAttributes() - 1);
        
        // deserialize model
        ois = new ObjectInputStream(
                           new FileInputStream("data/age_0_to_30.model"));
        cls = (Classifier) ois.readObject();
        ois.close();
        
        // evaluate classifier and print some statistics
        //Evaluation eval = new Evaluation(train);
        //eval.evaluateModel(cls, test);
        System.out.println("Chronic kidney disease:");
        //System.out.println(eval.toSummaryString("\nBase result\n===========\n", false));
        //System.out.println("Precision: " + eval.precision(test.numAttributes() - 1) + "\nRecall: " + eval.recall(test.numAttributes() - 1));
        
        
        Instances tmpTest;
        for(int i=5; ; i+=5) {
            if(i > test.numInstances()) {
                i = test.numInstances();
                
                eval = new Evaluation(train);
                eval.evaluateModel(cls, test);
                //System.out.println("Chronic kidney disease:");
                System.out.println(eval.toSummaryString("\nBase result\nnum of data " + i + "\n===========\n", false));
            
                rf = new Reframing();
                rf.reframing(train, test, cls, i);

                rq = new ReframingQuadratic();
                rq.reframing(train, test, cls, i);
                break;
            }
            tmpTest = new Instances(test);
            for(int j=tmpTest.numInstances()-1; j>=i; j--) tmpTest.delete(j);
            
            eval = new Evaluation(train);
            eval.evaluateModel(cls, tmpTest);
            //System.out.println("Chronic kidney disease:");
            System.out.println(eval.toSummaryString("\nBase result\nnum of data " + i + "\n===========\n", false));
            
            rf = new Reframing();
            rf.reframing(train, test, cls, i);

            rq = new ReframingQuadratic();
            rq.reframing(train, test, cls, i);
        }
    */    
        // retrain
        /*Instances retrain = new Instances(
                            new BufferedReader(
                                new FileReader("data/age_more_than_70_retrain.arff")));
        retrain.setClassIndex(test.numAttributes() - 1);
        
        cls = new NaiveBayes();
        cls.buildClassifier(retrain);
        
        eval = new Evaluation(retrain);
        eval.evaluateModel(cls, test);
        System.out.println(eval.toSummaryString("\nRetrain with 46 data\n===============\n", false));
        */
        
        /////////////////////////////////////////
        // Heart disease data
        /////////////////////////////////////////
        
        // load train data set
       /* train = new Instances(
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
        //eval = new Evaluation(train);
        //eval.evaluateModel(cls, test);
        System.out.println("Heart disease: (hungarian)");
        //System.out.println(eval.toSummaryString("\nResults\n======\n", false));
        */
    /*    
        Instances tmpTest;
        for(int i=30; ; i+=30) {
            if(i > test.numInstances()) {
                i = test.numInstances();
                
                eval = new Evaluation(train);
                eval.evaluateModel(cls, test);
                System.out.println(eval.toSummaryString("\nBase result\nnum of data " + i + "\n===========\n", false));
            
                rf = new Reframing();
                rf.reframing(train, test, cls, i);

                rq = new ReframingQuadratic();
                rq.reframing(train, test, cls, i);
                break;
            }
            tmpTest = new Instances(test);
            for(int j=tmpTest.numInstances()-1; j>=i; j--) tmpTest.delete(j);
            
            eval = new Evaluation(train);
            eval.evaluateModel(cls, tmpTest);
            
            System.out.println(eval.toSummaryString("\nBase result\nnum of data " + i + "\n===========\n", false));
            
            rf = new Reframing();
            rf.reframing(train, test, cls, i);

            rq = new ReframingQuadratic();
            rq.reframing(train, test, cls, i);
        }*/
        
        // retrain
        /*
        Instances retrain = new Instances(test);
        retrain.setClassIndex(retrain.numAttributes() - 1);
        cls = new NaiveBayes();
        cls.buildClassifier(retrain);
        Instances tmpRetrain;
        
        for(int i=30; ; i+=30){
            if(i > test.numInstances()) {
                i = test.numInstances();
                
                retrain = new Instances(test);
                retrain.setClassIndex(retrain.numAttributes() - 1);
                eval = new Evaluation(retrain);
                cls = new NaiveBayes();
                cls.buildClassifier(retrain);
                eval.evaluateModel(cls, test);
                System.out.println(eval.toSummaryString("\nretrain with " + i + " data" + "\n===========\n", false));
                break;
            }
            tmpRetrain = new Instances(test);
            for(int j=tmpRetrain.numInstances()-1; j>=i; j--) tmpRetrain.delete(j);
            
            eval = new Evaluation(tmpRetrain);
            cls = new NaiveBayes();
            cls.buildClassifier(tmpRetrain);
            eval.evaluateModel(cls, test);
            
            System.out.println(eval.toSummaryString("\nretrain with " + i + " data" + "\n===========\n", false));
        }*/
        
        
        /////////////////////////////////////////
        // Indian Liver Patient Dataset (ILPD)
        /////////////////////////////////////////
        
        // load train data set
        train = new Instances(
                            new BufferedReader(
                                new FileReader("data/ILDP_male.arff")));
        train.setClassIndex(train.numAttributes() - 1);
        
        // create model
        dp = new DataPreprocessor();
        dp.createModel("data/ILDP_male.arff", "data/ILDP_male.model");
        
        // load test data set
        test = new Instances(
                            new BufferedReader(
                                new FileReader("data/ILDP_female.arff")));
        test.setClassIndex(test.numAttributes() - 1);
        
        // deserialize model
        ois = new ObjectInputStream(
                           new FileInputStream("data/ILDP_male.model"));
        cls = (Classifier) ois.readObject();
        ois.close();
        
        // evaluate classifier and print some statistics
        //eval = new Evaluation(train);
        //eval.evaluateModel(cls, test);
        System.out.println("Indian Liver Patient Dataset (ILPD):");
        //System.out.println(eval.toSummaryString("\nResults\n======\n", false));
        
       /* Instances tmpTest;
        for(int i=15; ; i+=15) {
            if(i > test.numInstances()) {
                i = test.numInstances();
                
                eval = new Evaluation(train);
                eval.evaluateModel(cls, test);
                System.out.println(eval.toSummaryString("\nBase result\nnum of data " + i + "\n===========\n", false));
            
                rf = new Reframing();
                rf.reframing(train, test, cls, i);

                rq = new ReframingQuadratic();
                rq.reframing(train, test, cls, i);
                break;
            }
            tmpTest = new Instances(test);
            for(int j=tmpTest.numInstances()-1; j>=i; j--) tmpTest.delete(j);
            
            eval = new Evaluation(train);
            eval.evaluateModel(cls, tmpTest);
            
            System.out.println(eval.toSummaryString("\nBase result\nnum of data " + i + "\n===========\n", false));
            
            rf = new Reframing();
            rf.reframing(train, test, cls, i);

            rq = new ReframingQuadratic();
            rq.reframing(train, test, cls, i);
        }
        */
        // retrain
        
        Instances retrain = new Instances(test);
        retrain.setClassIndex(retrain.numAttributes() - 1);
        cls = new NaiveBayes();
        cls.buildClassifier(retrain);
        Instances tmpRetrain;
        
        for(int i=15; ; i+=15){
            if(i > test.numInstances()) {
                i = test.numInstances();
                
                retrain = new Instances(test);
                retrain.setClassIndex(retrain.numAttributes() - 1);
                eval = new Evaluation(retrain);
                cls = new NaiveBayes();
                cls.buildClassifier(retrain);
                eval.evaluateModel(cls, test);
                System.out.println(eval.toSummaryString("\nretrain with " + i + " data" + "\n===========\n", false));
                break;
            }
            tmpRetrain = new Instances(test);
            for(int j=tmpRetrain.numInstances()-1; j>=i; j--) tmpRetrain.delete(j);
            
            eval = new Evaluation(tmpRetrain);
            cls = new NaiveBayes();
            cls.buildClassifier(tmpRetrain);
            eval.evaluateModel(cls, test);
            
            System.out.println(eval.toSummaryString("\nretrain with " + i + " data" + "\n===========\n", false));
        }
        
        
        /////////////////////////////////////////
        // Synthetic data
        /////////////////////////////////////////
    /*    
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
        rf.reframing(train, test, cls, test.numInstances());
        
        rq = new ReframingQuadratic();
        rq.reframing(train, test, cls, test.numInstances());
    */
    }
    
}
