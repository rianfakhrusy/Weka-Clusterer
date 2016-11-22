package wekaclusterer;

import java.util.Random;
import java.util.Vector;
import weka.clusterers.RandomizableClusterer;
import weka.core.*;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;

public class myKMeans extends RandomizableClusterer {

  /**
   * Instansi dari training
   */
  private Instances m_instances;

  /**
   * menggantikan missing values
   */
  private ReplaceMissingValues m_ReplaceMissingFilter;

  /**
   * banyak cluster untuk digenerate
   */
  private int m_NumClusters = 2;

  /**
   * menangani centroid
   */
  private Instances m_ClusterCentroids;

  /**
   * variable cluster temporary
   */
  private int [] m_ClusterAssignments;

  /**
   * random seed
   */
  private int m_Seed = 10;

  /**
   * nilai minimum dari atribut
   */
  private double [] m_Min;
  
  /**
   * nilai maksimum dari atribut
   */
  private double [] m_Max;
  

  public void buildClusterer(Instances data) throws Exception {
    
    if (data.checkForStringAttributes()) {
      throw  new Exception("Attribut tidak dapat ditangani :(");
    }

    m_ReplaceMissingFilter = new ReplaceMissingValues();
    m_ReplaceMissingFilter.setInputFormat(data);
    m_instances = Filter.useFilter(data, m_ReplaceMissingFilter);

    m_Min = new double [m_instances.numAttributes()];
    m_Max = new double [m_instances.numAttributes()];
    for (int i = 0; i < m_instances.numAttributes(); i++) {
      m_Min[i] = m_Max[i] = Double.NaN;
    }

    for (int i = 0; i < m_instances.numInstances(); i++) {
      updateMinMax(m_instances.instance(i));
    }
    
    m_ClusterCentroids = new Instances(m_instances, m_NumClusters);
    m_ClusterAssignments = new int [m_instances.numInstances()];

    Random RandomO = new Random(m_Seed);

    for (int i = 0; i < m_NumClusters; i++) {
      int instIndex = Math.abs(RandomO.nextInt()) % m_instances.numInstances();
      m_ClusterCentroids.add(m_instances.instance(instIndex));
    }

    boolean converged = false;
    while (!converged) {
      converged = true;
      for (int i = 0; i < m_instances.numInstances(); i++) {
	Instance toCluster = m_instances.instance(i);
	int newC = clusterProcessedInstance(toCluster);
	if (newC != m_ClusterAssignments[i]) {
	  converged = false;
	}
	m_ClusterAssignments[i] = newC;
      }
      
      Instances [] tempI = new Instances[m_NumClusters];
      // update centroid
      m_ClusterCentroids = new Instances(m_instances, m_NumClusters);
      for (int i = 0; i < m_NumClusters; i++) {
	tempI[i] = new Instances(m_instances, 0);
      }
      for (int i = 0; i < m_instances.numInstances(); i++) {
	tempI[m_ClusterAssignments[i]].add(m_instances.instance(i));
      }
      for (int i = 0; i < m_NumClusters; i++) {
	double [] vals = new double[m_instances.numAttributes()];
	for (int j = 0; j < m_instances.numAttributes(); j++) {
	  vals[j] = tempI[i].meanOrMode(j);
	}
	m_ClusterCentroids.add(new Instance(1.0, vals));
      }
    }
  }


  private int clusterProcessedInstance(Instance instance) {
    double minDist = Integer.MAX_VALUE;
    int bestCluster = 0;
    for (int i = 0; i < m_NumClusters; i++) {
      double dist = distance(instance, m_ClusterCentroids.instance(i));
      if (dist < minDist) {
	minDist = dist;
	bestCluster = i;
      }
    }
    return bestCluster;
  }


  public int clusterInstance(Instance instance) throws Exception {
    m_ReplaceMissingFilter.input(instance);
    m_ReplaceMissingFilter.batchFinished();
    Instance inst = m_ReplaceMissingFilter.output();

    return clusterProcessedInstance(inst);
  }

      
  private double distance(Instance first, Instance second) {  

    double distance = 0;
    int firstI, secondI;

    for (int p1 = 0, p2 = 0; 
	 p1 < first.numValues() || p2 < second.numValues();) {
      if (p1 >= first.numValues()) {
	firstI = m_instances.numAttributes();
      } else {
	firstI = first.index(p1); 
      }
      if (p2 >= second.numValues()) {
	secondI = m_instances.numAttributes();
      } else {
	secondI = second.index(p2);
      }
      if (firstI == m_instances.classIndex()) {
	p1++; continue;
      } 
      if (secondI == m_instances.classIndex()) {
	p2++; continue;
      } 
      double diff;
      if (firstI == secondI) {
	diff = difference(firstI, 
			  first.valueSparse(p1),
			  second.valueSparse(p2));
	p1++; p2++;
      } else if (firstI > secondI) {
	diff = difference(secondI, 
			  0, second.valueSparse(p2));
	p2++;
      } else {
	diff = difference(firstI, 
			  first.valueSparse(p1), 0);
	p1++;
      }
      distance += diff * diff;
    }
    
    return Math.sqrt(distance / m_instances.numAttributes());
  }

  private double difference(int index, double val1, double val2) {

    switch (m_instances.attribute(index).type()) {
    case Attribute.NOMINAL:
      
      // kalau atribut nominal
      if (Instance.isMissingValue(val1) || 
	  Instance.isMissingValue(val2) ||
	  ((int)val1 != (int)val2)) {
	return 1;
      } else {
	return 0;
      }
    case Attribute.NUMERIC:

      // kalau atribut numerik
      if (Instance.isMissingValue(val1) || 
	  Instance.isMissingValue(val2)) {
	if (Instance.isMissingValue(val1) && 
	    Instance.isMissingValue(val2)) {
	  return 1;
	} else {
	  double diff;
	  if (Instance.isMissingValue(val2)) {
	    diff = norm(val1, index);
	  } else {
	    diff = norm(val2, index);
	  }
	  if (diff < 0.5) {
	    diff = 1.0 - diff;
	  }
	  return diff;
	}
      } else {
	return norm(val1, index) - norm(val2, index);
      }
    default:
      return 0;
    }
  }


  private double norm(double x, int i) {

    if (Double.isNaN(m_Min[i]) || Utils.eq(m_Max[i],m_Min[i])) {
      return 0;
    } else {
      return (x - m_Min[i]) / (m_Max[i] - m_Min[i]);
    }
  }


  private void updateMinMax(Instance instance) {  

    for (int j = 0;j < m_instances.numAttributes(); j++) {
      if (!instance.isMissing(j)) {
	if (Double.isNaN(m_Min[j])) {
	  m_Min[j] = instance.value(j);
	  m_Max[j] = instance.value(j);
	} else {
	  if (instance.value(j) < m_Min[j]) {
	    m_Min[j] = instance.value(j);
	  } else {
	    if (instance.value(j) > m_Max[j]) {
	      m_Max[j] = instance.value(j);
	    }
	  }
	}
      }
    }
  }
  

  public int numberOfClusters() throws Exception {
    return m_NumClusters;
  }
  

  public String numClustersTipText() {
    return "set number of clusters";
  }


  public void setNumClusters(int n) {
    m_NumClusters = n;
  }


  public int getNumClusters() {
    return m_NumClusters;
  }
    

  public String seedTipText() {
    return "random number seed";
  }



  public void setSeed (int s) {
    m_Seed = s;
  }



  public int getSeed () {
    return  m_Seed;
  }


  public String toString() {
    StringBuffer temp = new StringBuffer();

    temp.append("\nkMeans\n======\n");

    temp.append("\nCluster centroids:\n");
    for (int i = 0; i < m_NumClusters; i++) {
      temp.append("\nCluster "+i+"\n\t");
      for (int j = 0; j < m_ClusterCentroids.numAttributes(); j++) {
	if (m_ClusterCentroids.attribute(j).isNominal()) {
	  temp.append(" "+m_ClusterCentroids.attribute(j).
		      value((int)m_ClusterCentroids.instance(i).value(j)));
	} else {
	  temp.append(" "+m_ClusterCentroids.instance(i).value(j));
	}
      }
    }
    return temp.toString();
  }
}