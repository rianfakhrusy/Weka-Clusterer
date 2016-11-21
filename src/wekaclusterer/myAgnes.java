package wekaclusterer;

import java.io.Serializable;
import java.util.Comparator;
import java.util.Enumeration;
import java.util.PriorityQueue;
import java.util.Vector;
import weka.clusterers.AbstractClusterer;

import weka.core.Capabilities;
import weka.core.CapabilitiesHandler;
import weka.core.DistanceFunction;
import weka.core.EuclideanDistance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.OptionHandler;
import weka.core.SelectedTag;
import weka.core.Tag;
import weka.core.Utils;
import weka.core.Capabilities.Capability;

public class myAgnes extends AbstractClusterer implements OptionHandler, CapabilitiesHandler {

    Instances m_instances; //training data
    int m_nNumClusters = 2; //number of clusters

    /** distance function used for comparing members of a cluster **/
    protected DistanceFunction m_DistanceFunction = new EuclideanDistance();
    public DistanceFunction getDistanceFunction() {
        return m_DistanceFunction;
    }
    public void setDistanceFunction(DistanceFunction distanceFunction) {
        m_DistanceFunction = distanceFunction;
    }

    /** used for priority queue for efficient retrieval of pair of clusters to merge**/
    class Tuple {
      public Tuple(double d, int i, int j, int nSize1, int nSize2) {
        m_fDist = d;
        m_iCluster1 = i;
        m_iCluster2 = j;
        m_nClusterSize1 = nSize1;
        m_nClusterSize2 = nSize2;
      }
      double m_fDist;
      int m_iCluster1;
      int m_iCluster2;
      int m_nClusterSize1;
      int m_nClusterSize2;
    }
    /** comparator used by priority queue**/
    class TupleComparator implements Comparator<Tuple> {
      public int compare(Tuple o1, Tuple o2) {
        if (o1.m_fDist < o2.m_fDist) {
          return -1;
        } else if (o1.m_fDist == o2.m_fDist) {
          return 0;
        }
        return 1;
      }
    }

    /** the various link types */
    final static int SINGLE = 0;
    final static int COMPLETE = 1;
    public static final Tag[] TAGS_LINK_TYPE = {
      new Tag(SINGLE, "SINGLE"),
      new Tag(COMPLETE, "COMPLETE")
    };

    /**
     * Holds the Link type used calculate distance between clusters
     */
    int m_nLinkType = SINGLE;

    public void setLinkType(SelectedTag newLinkType) {
      if (newLinkType.getTags() == TAGS_LINK_TYPE) {
        m_nLinkType = newLinkType.getSelectedTag().getID();
      }
    }

    public SelectedTag getLinkType() {
      return new SelectedTag(m_nLinkType, TAGS_LINK_TYPE);
    }

    /** class representing node in cluster hierarchy **/
    class Node implements Serializable {
      Node m_left;
      Node m_right;
      Node m_parent;
      int m_iLeftInstance;
      int m_iRightInstance;
      double m_fLeftLength = 0;
      double m_fRightLength = 0;
      double m_fHeight = 0;
      
      void setHeight(double fHeight1, double fHeight2) {
        m_fHeight = fHeight1;
        if (m_left == null) {
          m_fLeftLength = fHeight1;
        } else {
          m_fLeftLength = fHeight1 - m_left.m_fHeight;
        }
        if (m_right == null) {
          m_fRightLength = fHeight2;
        } else {
          m_fRightLength = fHeight2 - m_right.m_fHeight;
        }
      }
      void setLength(double fLength1, double fLength2) {
        m_fLeftLength = fLength1;
        m_fRightLength = fLength2;
        m_fHeight = fLength1;
        if (m_left != null) {
          m_fHeight += m_left.m_fHeight;
        }
      }
    }
    Node [] m_clusters;
    int [] m_nClusterNr;


    @Override
    public void buildClusterer(Instances data) throws Exception {
      //		/System.err.println("Method " + m_nLinkType);
      m_instances = data;
      int nInstances = m_instances.numInstances();
      if (nInstances == 0) {
        return;
      }
      m_DistanceFunction.setInstances(m_instances);
      // use array of integer vectors to store cluster indices,
      // starting with one cluster per instance
      Vector<Integer> [] nClusterID = new Vector[data.numInstances()];
      for (int i = 0; i < data.numInstances(); i++) {
        nClusterID[i] = new Vector<Integer>();
        nClusterID[i].add(i);
      }
      // calculate distance matrix
      int nClusters = data.numInstances();

      // used for keeping track of hierarchy
      Node [] clusterNodes = new Node[nInstances];
      
      doLinkClustering(nClusters, nClusterID, clusterNodes);

      // move all clusters in m_nClusterID array
      // & collect hierarchy
      int iCurrent = 0;
      m_clusters = new Node[m_nNumClusters];
      m_nClusterNr = new int[nInstances];
      for (int i = 0; i < nInstances; i++) {
        if (nClusterID[i].size() > 0) {
          for (int j = 0; j < nClusterID[i].size(); j++) {
            m_nClusterNr[nClusterID[i].elementAt(j)] = iCurrent;
          }
          m_clusters[iCurrent] = clusterNodes[i];
          iCurrent++;
        }
      }

    } // buildClusterer

    /** Perform clustering using a link method
     * This implementation uses a priority queue resulting in a O(n^2 log(n)) algorithm
     * @param nClusters number of clusters
     * @param nClusterID 
     * @param clusterNodes 
     */
    void doLinkClustering(int nClusters, Vector<Integer>[] nClusterID, Node [] clusterNodes) {
      int nInstances = m_instances.numInstances();
      PriorityQueue<Tuple> queue = new PriorityQueue<Tuple>(nClusters*nClusters/2, new TupleComparator());
      double [][] fDistance0 = new double[nClusters][nClusters];
      double [][] fClusterDistance = null;

      for (int i = 0; i < nClusters; i++) {
        fDistance0[i][i] = 0;
        for (int j = i+1; j < nClusters; j++) {
          fDistance0[i][j] = getDistance0(nClusterID[i], nClusterID[j]);
          fDistance0[j][i] = fDistance0[i][j];
          queue.add(new Tuple(fDistance0[i][j], i, j, 1, 1));
        }
      }
      while (nClusters > m_nNumClusters) {
        int iMin1 = -1;
        int iMin2 = -1;
        // find closest two clusters
       
        // use priority queue to find next best pair to cluster
        Tuple t;
        do {
          t = queue.poll();
        } while (t!=null && (nClusterID[t.m_iCluster1].size() != t.m_nClusterSize1 || nClusterID[t.m_iCluster2].size() != t.m_nClusterSize2));
        iMin1 = t.m_iCluster1;
        iMin2 = t.m_iCluster2;
        merge(iMin1, iMin2, t.m_fDist, t.m_fDist, nClusterID, clusterNodes);
        
        // merge  clusters

        // update distances & queue
        for (int i = 0; i < nInstances; i++) {
          if (i != iMin1 && nClusterID[i].size()!=0) {
            int i1 = Math.min(iMin1,i);
            int i2 = Math.max(iMin1,i);
            double fDistance = getDistance(fDistance0, nClusterID[i1], nClusterID[i2]);
            
            queue.add(new Tuple(fDistance, i1, i2, nClusterID[i1].size(), nClusterID[i2].size()));
          }
        }

        nClusters--;
      }
    } // doLinkClustering

    void merge(int iMin1, int iMin2, double fDist1, double fDist2, Vector<Integer>[] nClusterID, Node [] clusterNodes) {
      if (iMin1 > iMin2) {
        int h = iMin1; iMin1 = iMin2; iMin2 = h;
        double f = fDist1; fDist1 = fDist2; fDist2 = f;
      }
      nClusterID[iMin1].addAll(nClusterID[iMin2]);
      nClusterID[iMin2].removeAllElements();

      // track hierarchy
      Node node = new Node();
      if (clusterNodes[iMin1] == null) {
        node.m_iLeftInstance = iMin1;
      } else {
        node.m_left = clusterNodes[iMin1];
        clusterNodes[iMin1].m_parent = node;
      }
      if (clusterNodes[iMin2] == null) {
        node.m_iRightInstance = iMin2;
      } else {
        node.m_right = clusterNodes[iMin2];
        clusterNodes[iMin2].m_parent = node;
      }
      
      node.setHeight(fDist1, fDist2);
      
      clusterNodes[iMin1] = node;
    } // merge

    /** calculate distance the first time when setting up the distance matrix **/
    double getDistance0(Vector<Integer> cluster1, Vector<Integer> cluster2) {
        double fBestDist = Double.MAX_VALUE;
        
        // set up two instances for distance function
        Instance instance1 = (Instance) m_instances.instance(cluster1.elementAt(0)).copy();
        Instance instance2 = (Instance) m_instances.instance(cluster2.elementAt(0)).copy();
        fBestDist = m_DistanceFunction.distance(instance1, instance2);
        
        return fBestDist;
    } // getDistance0

    /** calculate the distance between two clusters 
     * @param cluster1 list of indices of instances in the first cluster
     * @param cluster2 dito for second cluster
     * @return distance between clusters based on link type
     */
    double getDistance(double [][] fDistance, Vector<Integer> cluster1, Vector<Integer> cluster2) {
      double fBestDist = Double.MAX_VALUE;
      switch (m_nLinkType) {
      case SINGLE:
        // find single link distance aka minimum link, which is the closest distance between
        // any item in cluster1 and any item in cluster2
        fBestDist = Double.MAX_VALUE;
        for (int i = 0; i < cluster1.size(); i++) {
          int i1 = cluster1.elementAt(i);
          for (int j = 0; j < cluster2.size(); j++) {
            int i2  = cluster2.elementAt(j);
            double fDist = fDistance[i1][i2];
            if (fBestDist > fDist) {
              fBestDist = fDist;
            }
          }
        }
        break;
      case COMPLETE:
        // find complete link distance aka maximum link, which is the largest distance between
        // any item in cluster1 and any item in cluster2
        fBestDist = 0;
        for (int i = 0; i < cluster1.size(); i++) {
          int i1 = cluster1.elementAt(i);
          for (int j = 0; j < cluster2.size(); j++) {
            int i2 = cluster2.elementAt(j);
            double fDist = fDistance[i1][i2];
            if (fBestDist < fDist) {
              fBestDist = fDist;
            }
          }
        }
        if (m_nLinkType == COMPLETE) {
          break;
        }
        // calculate adjustment, which is the largest within cluster distance
        double fMaxDist = 0;
        for (int i = 0; i < cluster1.size(); i++) {
          int i1 = cluster1.elementAt(i);
          for (int j = i+1; j < cluster1.size(); j++) {
            int i2 = cluster1.elementAt(j);
            double fDist = fDistance[i1][i2];
            if (fMaxDist < fDist) {
              fMaxDist = fDist;
            }
          }
        }
        for (int i = 0; i < cluster2.size(); i++) {
          int i1 = cluster2.elementAt(i);
          for (int j = i+1; j < cluster2.size(); j++) {
            int i2 = cluster2.elementAt(j);
            double fDist = fDistance[i1][i2];
            if (fMaxDist < fDist) {
              fMaxDist = fDist;
            }
          }
        }
        fBestDist -= fMaxDist;
        break;
      }
      return fBestDist;
    } // getDistance

    /** calculated error sum-of-squares for instances wrt centroid **/
    double calcESS(Vector<Integer> cluster) {
      double [] fValues1 = new double[m_instances.numAttributes()];
      for (int i = 0; i < cluster.size(); i++) {
        Instance instance = m_instances.instance(cluster.elementAt(i));
        for (int j = 0; j < m_instances.numAttributes(); j++) {
          fValues1[j] += instance.value(j);
        }
      }
      for (int j = 0; j < m_instances.numAttributes(); j++) {
        fValues1[j] /= cluster.size();
      }
      // set up two instances for distance function
      Instance centroid = (Instance) m_instances.instance(cluster.elementAt(0)).copy();
      for (int j = 0; j < m_instances.numAttributes(); j++) {
        centroid.setValue(j, fValues1[j]);
      }
      double fESS = 0;
      for (int i = 0; i < cluster.size(); i++) {
        Instance instance = m_instances.instance(cluster.elementAt(i));
        fESS += m_DistanceFunction.distance(centroid, instance);
      }
      return fESS / cluster.size(); 
    } // calcESS

    @Override
    /** instances are assigned a cluster by finding the instance in the training data 
     * with the closest distance to the instance to be clustered. The cluster index of
     * the training data point is taken as the cluster index.
     */
    public int clusterInstance(Instance instance) throws Exception {
      if (m_instances.numInstances() == 0) {
        return 0;
      }
      double fBestDist = Double.MAX_VALUE;
      int iBestInstance = -1;
      for (int i = 0; i < m_instances.numInstances(); i++) {
        double fDist = m_DistanceFunction.distance(instance, m_instances.instance(i));
        if (fDist < fBestDist) {
          fBestDist = fDist;
          iBestInstance = i;
        }
      }
      return m_nClusterNr[iBestInstance];
    }

    @Override
    /** create distribution with all clusters having zero probability, except the
     * cluster the instance is assigned to.
     */
    public double[] distributionForInstance(Instance instance) throws Exception {
      if (numberOfClusters() == 0) {
        double [] p = new double[1];
        p[0] = 1;
        return p;
      }
      double [] p = new double[numberOfClusters()];
      p[clusterInstance(instance)] = 1.0;
      return p;
    }

    @Override
    public Capabilities getCapabilities() {
      Capabilities result = new Capabilities(this);
      result.disableAll();
      result.enable(Capability.NO_CLASS);

      // attributes
      result.enable(Capability.NOMINAL_ATTRIBUTES);
      result.enable(Capability.NUMERIC_ATTRIBUTES);
      result.enable(Capability.DATE_ATTRIBUTES);
      result.enable(Capability.MISSING_VALUES);
      result.enable(Capability.STRING_ATTRIBUTES);

      // other
      result.setMinimumNumberInstances(0);
      return result;
    }

    @Override
    public int numberOfClusters() throws Exception {
      return Math.min(m_nNumClusters, m_instances.numInstances());
    }

    /**
     * Returns an enumeration describing the available options.
     *
     * @return an enumeration of all the available options.
     */
    public Enumeration listOptions() {

      Vector newVector = new Vector(8);
      newVector.addElement(new Option(
          "\tIf set, classifier is run in debug mode and\n"
          + "\tmay output additional info to the console",
          "D", 0, "-D"));
      newVector.addElement(new Option(
          "\tIf set, distance is interpreted as branch length\n"
          + "\totherwise it is node height.",
          "B", 0, "-B"));

      newVector.addElement(new Option(
          "\tnumber of clusters",
          "N", 1,"-N <Nr Of Clusters>"));
      newVector.addElement(new Option(
          "\tFlag to indicate the cluster should be printed in Newick format.",
          "P", 0,"-P"));
      newVector.addElement(
          new Option(
              "Link type (Single, Complete, Average, Mean, Centroid, Ward, Adjusted complete, Neighbor joining)", "L", 1,
          "-L [SINGLE|COMPLETE|AVERAGE|MEAN|CENTROID|WARD|ADJCOMLPETE|NEIGHBOR_JOINING]"));
      newVector.add(new Option(
          "\tDistance function to use.\n"
          + "\t(default: weka.core.EuclideanDistance)",
          "A", 1,"-A <classname and options>"));
      return newVector.elements();
    }

    /**
     * Parses a given list of options. <p/>
     *
             <!-- options-start -->
     * Valid options are: <p/>
     * 
             <!-- options-end -->
     *
     * @param options the list of options as an array of strings
     * @throws Exception if an option is not supported
     */
    public void setOptions(String[] options) throws Exception {

      String optionString = Utils.getOption('N', options); 
      if (optionString.length() != 0) {
        Integer temp = new Integer(optionString);
        setNumClusters(temp);
      }
      else {
        setNumClusters(2);
      }

      String sLinkType = Utils.getOption('L', options);


      if (sLinkType.compareTo("SINGLE") == 0) {setLinkType(new SelectedTag(SINGLE, TAGS_LINK_TYPE));}
      if (sLinkType.compareTo("COMPLETE") == 0) {setLinkType(new SelectedTag(COMPLETE, TAGS_LINK_TYPE));}

      String nnSearchClass = Utils.getOption('A', options);
      if(nnSearchClass.length() != 0) {
        String nnSearchClassSpec[] = Utils.splitOptions(nnSearchClass);
        if(nnSearchClassSpec.length == 0) { 
          throw new Exception("Invalid DistanceFunction specification string."); 
        }
        String className = nnSearchClassSpec[0];
        nnSearchClassSpec[0] = "";

        setDistanceFunction( (DistanceFunction)
            Utils.forName( DistanceFunction.class, 
                className, nnSearchClassSpec) );
      }
      else {
        setDistanceFunction(new EuclideanDistance());
      }

      Utils.checkForRemainingOptions(options);
    }

    /**
     * Gets the current settings of the clusterer.
     *
     * @return an array of strings suitable for passing to setOptions()
     */
    public String [] getOptions() {

      String [] options = new String [14];
      int current = 0;

      options[current++] = "-N";
      options[current++] = "" + getNumClusters();

      options[current++] = "-L";
      switch (m_nLinkType) {
      case (SINGLE) :options[current++] = "SINGLE";break;
      case (COMPLETE) :options[current++] = "COMPLETE";break;
      }

      options[current++] = "-A";
      options[current++] = (m_DistanceFunction.getClass().getName() + " " +
          Utils.joinOptions(m_DistanceFunction.getOptions())).trim();

      while (current < options.length) {
        options[current++] = "";
      }

      return options;
    }
    
    //setter
    public void setNumClusters(int nClusters) {
        m_nNumClusters = Math.max(1,nClusters);
    }
    
    //getter
    public int getNumClusters() {
        return m_nNumClusters;
    }
}
