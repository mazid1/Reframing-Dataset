package reframing;

import java.io.FileReader;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.Vector;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

/**
 *
 * @author Mazid
 */
public class DataPreprocessor {
                
    public void preprocess(String src, String dest) throws IOException {
        // load data set
        Instances instances = new Instances(
                            new BufferedReader(
                                new FileReader(src)));
        
        for(int i=23; i>=11; i--)
            instances.deleteAttributeAt(i);
        
        for(int i=8; i>=2; i--)
            instances.deleteAttributeAt(i);
        
        for(int i=0; i<=4; i++)
            instances.deleteWithMissing(i);
        
        BufferedWriter writer = new BufferedWriter(new FileWriter(dest));
        writer.write(instances.toString());
        writer.flush();
        writer.close();
    }
    
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
