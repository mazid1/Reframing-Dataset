package reframing;

import java.io.FileReader;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.ObjectOutputStream;

import weka.classifiers.Classifier;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instances;

/**
 *
 * @author Mazid
 */
public class DataPreprocessor {
    
    // remove some unecessary attributes and instaces with missing values
    public void preprocess(String src, String dest) throws IOException {
        // load data set
        Instances instances = new Instances(
                            new BufferedReader(
                                new FileReader(src)));
        
        for(int i=23; i>=11; i--)
            instances.deleteAttributeAt(i);
        
        for(int i=8; i>=2; i--)
            instances.deleteAttributeAt(i);
        
        //for(int i=0; i<=4; i++)
            //instances.deleteWithMissing(i);
        instances.deleteWithMissing(0); // remove instances with missing age
        
        BufferedWriter writer = new BufferedWriter(new FileWriter(dest));
        writer.write(instances.toString());
        writer.flush();
        writer.close();
    }
    
    // generate new arff file for "age" value from 0 to 30
    public void age_0_to_30(String src, String dest) throws IOException {
        // load data set
        Instances instances = new Instances(
                            new BufferedReader(
                                new FileReader(src)));
        
        int numOfInstances = instances.numInstances();
        
        // remove instances with age > 30
        for(int i=numOfInstances-1; i>=0; i--) {
            double age = (double) instances.get(i).value(0);
            if(age > 30.0) instances.delete(i);
        }
        
        // write in new file
        BufferedWriter writer = new BufferedWriter(new FileWriter(dest));
        writer.write(instances.toString());
        writer.flush();
        writer.close();
    }
    
    // generate new arff file for "age" value greater than 70
    public void age_greater_than_70(String src, String dest) throws IOException {
        // load data set
        Instances instances = new Instances(
                            new BufferedReader(
                                new FileReader(src)));
        
        int numOfInstances = instances.numInstances();
        
        // remove instances with age <= 70
        for(int i=numOfInstances-1; i>=0; i--) {
            double age = (double) instances.get(i).value(0);
            if(age <= 70.0) instances.delete(i);
        }
        
        // write in new file
        BufferedWriter writer = new BufferedWriter(new FileWriter(dest));
        writer.write(instances.toString());
        writer.flush();
        writer.close();
    }
    
    // create new model using train data
    public void createModel(String src, String dest) throws Exception {
        // create NaiveBayes
        Classifier cls = new NaiveBayes();

        // train
        Instances inst = new Instances(
                           new BufferedReader(
                             new FileReader(src)));
        inst.setClassIndex(inst.numAttributes() - 1);
        cls.buildClassifier(inst);

        // serialize model
        ObjectOutputStream oos = new ObjectOutputStream(
                                   new FileOutputStream(dest));
        oos.writeObject(cls);
        oos.flush();
        oos.close();
    }
    
}
