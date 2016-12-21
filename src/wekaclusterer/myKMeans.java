package wekaclusterer;

import java.util.ArrayList;
import java.util.Random;
import weka.clusterers.RandomizableClusterer;
import weka.core.*;
import weka.core.Capabilities.Capability;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;

public class myKMeans extends RandomizableClusterer {
    private Instances m_instances; //training data
    private ReplaceMissingValues m_ReplaceMissingFilter; //for replacement of missing value
    private int m_NumClusters; //number of clusters
    private Instances m_ClusterCentroids; //clusters centroid
    private int m_Seed; //randomization seed
    private ArrayList<ArrayList<Integer>> nClusterID; //storing cluster ID
    private int m_Iterations; //number of iterations before convergence
    private double[] m_Min; //store minimum values for all attributes
    private double[] m_Max; //store maximum values for all attributes
    protected double[] m_squaredErrors;
    
    //constructor
    public myKMeans() {
        this.nClusterID = new ArrayList<>();
        this.m_NumClusters = 2;
        this.m_Seed = 10;
        this.m_Iterations = 0;
    }
    
    //capability handler
    @Override
    public Capabilities getCapabilities() {
        Capabilities result = super.getCapabilities();
        result.disableAll();
        //class
        result.enable(Capability.NO_CLASS);
        result.enable(Capability.MISSING_VALUES);
        // attributes
        result.enable(Capability.NOMINAL_ATTRIBUTES);
        result.enable(Capability.NUMERIC_ATTRIBUTES);
        return result;
    }
    
    @Override
    public void buildClusterer(Instances data) throws Exception {
        //replace instance with missing values
        m_ReplaceMissingFilter = new ReplaceMissingValues();
        m_ReplaceMissingFilter.setInputFormat(data);
        m_instances = Filter.useFilter(data, m_ReplaceMissingFilter);
         m_squaredErrors = new double[m_NumClusters];
         
        //initialize minimum and maximum value of attributes
        m_Min = new double[m_instances.numAttributes()];
        m_Max = new double[m_instances.numAttributes()];
        for (int i = 0; i < m_instances.numAttributes(); i++)
            m_Min[i] = m_Max[i] = Double.NaN;
        
        //updates minimum and maximum values for all attributes based on new instance
        for (int i = 0; i < m_instances.numInstances(); i++){
            for (int j = 0;j < m_instances.numAttributes(); j++){
                Instance instance = m_instances.instance(i);
                if (!instance.isMissing(j))
                    if (Double.isNaN(m_Min[j])) {
                        m_Min[j] = instance.value(j);
                        m_Max[j] = instance.value(j);
                    } else if (instance.value(j) < m_Min[j])
                        m_Min[j] = instance.value(j);
                    else if (instance.value(j) > m_Max[j])
                        m_Max[j] = instance.value(j);
            }
        }
    
        m_ClusterCentroids = new Instances(m_instances, m_NumClusters); //temp for centroids
        int[] m_ClusterAssignments = new int[m_instances.numInstances()]; //temp variable for cluster assignment
        
        //select initial centroid at random
        Random random = new Random(m_Seed);
        for (int i = 0; i < m_NumClusters; i++) {
            int instIndex = Math.abs(random.nextInt()) % m_instances.numInstances();
            m_ClusterCentroids.add(m_instances.instance(instIndex));
        }
        
        //begin iterations until converged
        boolean converged = false;
        while (!converged) {
            m_squaredErrors = new double[m_NumClusters];
            m_Iterations++; //iterations increment
            converged = true;
            //assign instance to a cluster
            for (int i = 0; i < m_instances.numInstances(); i++) {
                Instance toCluster = m_instances.instance(i);
                int newC = clusterInstance(toCluster);
                if (newC != m_ClusterAssignments[i])
                    converged = false;
                m_ClusterAssignments[i] = newC;
            }
            
            Instances[] temp = new Instances[m_NumClusters];
            m_ClusterCentroids = new Instances(m_instances, m_NumClusters); 
            for (int i = 0; i < m_NumClusters; i++)
                temp[i] = new Instances(m_instances, 0);
            //adding instance to cluster
            for (int i = 0; i < m_instances.numInstances(); i++)
                temp[m_ClusterAssignments[i]].add(m_instances.instance(i));
            //update centroids
            for (int i = 0; i < m_NumClusters; i++) {
                double[] vals = new double[m_instances.numAttributes()];
                for (int j = 0; j < m_instances.numAttributes(); j++)
                  vals[j] = temp[i].meanOrMode(j);
                m_ClusterCentroids.add(new Instance(1.0, vals)); 
            }
        }
        //record the member of clusters
        for (int i = 0; i < numberOfClusters(); i++){
            nClusterID.add(new ArrayList<>());
        }
        for (int i = 0; i < m_instances.numInstances(); i++){
            nClusterID.get(m_ClusterAssignments[i]).add(i);
        }
    }

    //assign a given instance to a cluster
    @Override
    public int clusterInstance(Instance instance) throws Exception {
        //replace missing value
        m_ReplaceMissingFilter.input(instance);
        m_ReplaceMissingFilter.batchFinished();
        Instance inst = m_ReplaceMissingFilter.output();
        //cluster instance
        double minDist = Integer.MAX_VALUE; //minimum distance initialization
        int bestCluster = 0; //default cluster
        //search best cluster for an instance
        for (int i = 0; i < m_NumClusters; i++) {
            double dist = distance(inst, m_ClusterCentroids.instance(i));
            if (dist < minDist) {
                minDist = dist;
                bestCluster = i;
            }
        }
        minDist *= minDist;
        m_squaredErrors[bestCluster] += minDist;
        return bestCluster;
    }

    //calculates the distance between two instances
    private double distance(Instance first, Instance second) {  
        double distance = 0;
        int firstI, secondI;

        for (int p1 = 0, p2 = 0; p1 < first.numValues() || p2 < second.numValues();) {
            firstI = (p1 >= first.numValues())? m_instances.numAttributes(): first.index(p1);
            secondI = (p2 >= second.numValues())? m_instances.numAttributes(): second.index(p2); 
            if (firstI == m_instances.classIndex())
                p1++; 
            else if (secondI == m_instances.classIndex())
                p2++;
            else {
                double diff;
                if (firstI == secondI) {
                    diff = difference(firstI,first.valueSparse(p1),second.valueSparse(p2));
                    p1++; 
                    p2++;
                } else if (firstI > secondI) {
                    diff = difference(secondI,0, second.valueSparse(p2));
                    p2++;
                } else {
                    diff = difference(firstI,first.valueSparse(p1), 0);
                    p1++;
                }
                distance += diff * diff;
            }
        }

        return Math.sqrt(distance / m_instances.numAttributes());
    }

    //count the distance between two attributes values
    private double difference(int index, double val1, double val2) {
        switch (m_instances.attribute(index).type()) {
            case Attribute.NOMINAL: //if attribute is nominal, check if there is missing value
                if (Instance.isMissingValue(val1) || Instance.isMissingValue(val2) || ((int)val1 != (int)val2))
                    return 1;
                else
                    return 0;
            case Attribute.NUMERIC: //if attribute is numeric, check if there is missing values and normalize
                if (Instance.isMissingValue(val1) || Instance.isMissingValue(val2)) { 
                    if (Instance.isMissingValue(val1) &&  Instance.isMissingValue(val2))
                        return 1;
                    else {
                        double diff;
                        if (Instance.isMissingValue(val2))
                            diff = normalize(val1, index);
                        else 
                            diff = normalize(val2, index);
                        if (diff < 0.5) 
                            diff = 1.0 - diff;
                        return diff;
                    }
                } else
                    return normalize(val1, index) - normalize(val2, index);
            default:
                return 0;
        }
    }

    //normalize numeric attribute.
    private double normalize(double x, int i) {
        if (Double.isNaN(m_Min[i]) || Utils.eq(m_Max[i],m_Min[i]))
            return 0;
        else
            return (x - m_Min[i]) / (m_Max[i] - m_Min[i]);
    }
  
    //return the number of clusters
    @Override
    public int numberOfClusters() throws Exception {
        return Math.min(m_NumClusters, m_instances.numInstances());
    }

    //getter
    //get the random number seed
    @Override
    public int getSeed () {
        return  m_Seed;
    }
    
    // gets the number of clusters
    public int getNumClusters() {
        return m_NumClusters;
    }
    
    //setter
    //set the random number seed
    @Override
    public void setSeed (int s) {
        m_Seed = s;
    }
    
    //set the number of clusters
    public void setNumClusters(int n) {
        m_NumClusters = n;
    }

    //info about the clusters
    @Override
    public String toString() {
        StringBuilder temp = new StringBuilder();
        temp.append("Within cluster sum of squared errors: "
        + Utils.sum(m_squaredErrors));
        temp.append("\nkMeans\n======\n")
                .append("\nNumber of iterations: ").append(m_Iterations).append("\n")
                .append("\nCluster members:\n\n");
        //output instances in all cluster
        for (int i = 0; i < nClusterID.size(); i++)
            temp.append("Cluster ").append(i).append(": ").append(nClusterID.get(i).toString()).append("\n");
        //output all clusters centroid
        temp.append("\nCluster centroids:\n");
        for (int i = 0; i < m_NumClusters; i++) {
            temp.append("\nCluster ").append(i).append(" ");
            for (int j = 0; j < m_ClusterCentroids.numAttributes(); j++) {
                if (m_ClusterCentroids.attribute(j).isNominal()) {
                    temp.append(" ").append(m_ClusterCentroids.attribute(j).
                            value((int)m_ClusterCentroids.instance(i).value(j)));
                } else
                    temp.append(" ").append(m_ClusterCentroids.instance(i).value(j));
              }
        }
        temp.append("\n\n");
        return temp.toString();
    }
}