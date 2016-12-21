package wekaclusterer;

import java.util.Comparator;
import java.util.PriorityQueue;
import java.util.ArrayList;
import weka.clusterers.AbstractClusterer;
import weka.core.Capabilities;
import weka.core.EuclideanDistance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Capabilities.Capability;

public class myAgnes extends AbstractClusterer{

    // inner class used for priority queue of merging clusters
    class Tuple {
        double m_fDist;
        int m_iCluster1;
        int m_iCluster2;
        int m_nClusterSize1;
        int m_nClusterSize2;
        public Tuple(double d, int i, int j, int nSize1, int nSize2) {
            m_fDist = d;
            m_iCluster1 = i;
            m_iCluster2 = j;
            m_nClusterSize1 = nSize1;
            m_nClusterSize2 = nSize2;
        }
    }
    // comparator used by priority queue
    class TupleComparator implements Comparator<Tuple> {
        @Override
        public int compare(Tuple o1, Tuple o2) {
            if (o1.m_fDist < o2.m_fDist) 
                return -1;
            else if (o1.m_fDist == o2.m_fDist)
                return 0;
            return 1;
        }
    }
    // nodes in cluster hierachy
    class Node {
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
            m_fLeftLength = (m_left == null)? fHeight1: fHeight1 - m_left.m_fHeight;
            m_fRightLength = (m_right == null)? fHeight2: fHeight2 - m_right.m_fHeight;
        }
        void setLength(double fLength1, double fLength2) {
            m_fLeftLength = fLength1;
            m_fRightLength = fLength2;
            m_fHeight = fLength1;
            if (m_left != null) 
                m_fHeight += m_left.m_fHeight;
        }
    }
    
    //link types to calculate distance between a pair of clusters
    final static int SINGLE = 0; 
    final static int COMPLETE = 1;
    int m_nLinkType = SINGLE;
    
    Instances m_instances; //training data
    int m_nNumClusters = 2; //default number of clusters
    EuclideanDistance m_EuclideanDistance = new EuclideanDistance(); //distance function
    Node[] m_clusters; //tree structures to store cluster hierarchy
    int[] m_nClusterNr; //number/ID of a cluster
    ArrayList<Integer>[] nClusterID; //storing cluster ID, one cluster/instance at first
    
    //capability handler
    @Override
    public Capabilities getCapabilities() {
        Capabilities capabilities = new Capabilities(this);
        capabilities.disableAll();
        //class
        capabilities.enable(Capability.NO_CLASS);
        capabilities.enable(Capability.MISSING_VALUES);
        // attributes
        capabilities.enable(Capability.NOMINAL_ATTRIBUTES);
        capabilities.enable(Capability.NUMERIC_ATTRIBUTES);
        capabilities.enable(Capability.DATE_ATTRIBUTES);
        capabilities.enable(Capability.STRING_ATTRIBUTES);
        // instances
        capabilities.setMinimumNumberInstances(0);
        return capabilities;
    }
    
    @Override
    public void buildClusterer(Instances data){
        m_instances = data;
        int nInstances = m_instances.numInstances(); //number of dataset
        if (nInstances == 0) //if there is no data
            return;
        m_EuclideanDistance.setInstances(m_instances);
        nClusterID = new ArrayList[data.numInstances()]; //storing cluster ID, one cluster/instance at first
        for (int i = 0; i < data.numInstances(); i++) {
            nClusterID[i] = new ArrayList<>();
            nClusterID[i].add(i);
        }
        Node[] clusterNodes = new Node[nInstances]; //temp for tree structures to store cluster hierarchy 

        doLinkClustering(nInstances, clusterNodes); //begin clustering
        
        //copy the clusters hierarchy from nclusterID array 
        int iCurrent = 0;
        m_clusters = new Node[m_nNumClusters]; //new tree structure
        m_nClusterNr = new int[nInstances]; //new clusters number
        for (int i = 0; i < nInstances; i++)
            if (nClusterID[i].size() > 0) {
                for (int j = 0; j < nClusterID[i].size(); j++)
                    m_nClusterNr[nClusterID[i].get(j)] = iCurrent;
                m_clusters[iCurrent] = clusterNodes[i];
                iCurrent++;
            }
    }

    //begin clustering
    void doLinkClustering(int nClusters, Node[] clusterNodes) {
        int nInstances = m_instances.numInstances(); //number of dataset instances
        PriorityQueue<Tuple> queue = new PriorityQueue<>(nClusters*nClusters/2, new TupleComparator());
        double[][] fDistance = new double[nClusters][nClusters]; //distance matrix
        
        //calculate all the distance and copy to a distance matrix
        for (int i = 0; i < nClusters; i++) 
            for (int j = i+1; j < nClusters; j++) {
                Instance instance1 = (Instance) m_instances.instance(nClusterID[i].get(0)).copy();
                Instance instance2 = (Instance) m_instances.instance(nClusterID[j].get(0)).copy();
                
                fDistance[i][j] = m_EuclideanDistance.distance(instance1,instance2);
                fDistance[j][i] = fDistance[i][j];
                queue.add(new Tuple(fDistance[i][j], i, j, 1, 1));
            }
        
        //merging clusters until there is only specified number of clusters left
        while (nClusters > m_nNumClusters) {
            //priority queue is used to find the closest pair of clusters
            Tuple t;
            do 
                t = queue.poll();
            while (t!=null && (nClusterID[t.m_iCluster1].size() != t.m_nClusterSize1 || nClusterID[t.m_iCluster2].size() != t.m_nClusterSize2));
            
            //merging two clusters with closest distance
            int iMin1 = t.m_iCluster1; 
            int iMin2 = t.m_iCluster2;
            merge(iMin1, iMin2, t.m_fDist, t.m_fDist, clusterNodes); 

            //updating distances & queue
            for (int i = 0; i < nInstances; i++) {
                if (i != iMin1 && !nClusterID[i].isEmpty()) {
                    int i1 = Math.min(iMin1,i);
                    int i2 = Math.max(iMin1,i);
                    queue.add(new Tuple(getDistance(fDistance, nClusterID[i1], nClusterID[i2]), i1, i2, nClusterID[i1].size(), nClusterID[i2].size()));
                }
            }
            nClusters--; //reduce the number of clusters
        }
    }
    
    //merge two clusters into one
    void merge(int iMin1, int iMin2, double fDist1, double fDist2, Node[] clusterNodes) {
        //merge two clusters into one
        nClusterID[iMin1].addAll(nClusterID[iMin2]);
        nClusterID[iMin2].clear();
        //debug System.out.println(nClusterID[iMin1].toString()); 

        //appends new cluster node to the tree
        Node node = new Node();
        if (clusterNodes[iMin1] == null)
            node.m_iLeftInstance = iMin1;
        else {
            node.m_left = clusterNodes[iMin1];
            clusterNodes[iMin1].m_parent = node;
        }
        node.m_iRightInstance = iMin2;
        node.setHeight(fDist1, fDist2); //update tree hieght
        clusterNodes[iMin1] = node;
    }

    //get the closest distance between two clusters based on link type from distance matrix
    double getDistance(double[][] fDistance, ArrayList<Integer> cluster1, ArrayList<Integer> cluster2) {
        double fBestDist = m_nLinkType==SINGLE?Double.MAX_VALUE:0; //set the default value of best distance based on link type
        for (int i = 0; i < cluster1.size(); i++) {
            int i1 = cluster1.get(i);
            for (int j = 0; j < cluster2.size(); j++) {
                int i2  = cluster2.get(j);
                double fDist = fDistance[i1][i2];
                if ((m_nLinkType==SINGLE)&&(fBestDist > fDist))
                    fBestDist = fDist;
                else if ((m_nLinkType==COMPLETE)&&(fBestDist < fDist))
                    fBestDist = fDist;
            }
        }
        return fBestDist;
    }
    
    //return id of an instance with the closest distance to the input instance to be clustered
    @Override
    public int clusterInstance(Instance instance){
        if (m_instances.numInstances() == 0)
            return 0;
        double fBestDist = Double.MAX_VALUE;
        int iBestInstance = -1;
        for (int i = 0; i < m_instances.numInstances(); i++) {
            double fDist = m_EuclideanDistance.distance(instance, m_instances.instance(i));
            if (fDist < fBestDist) {
                fBestDist = fDist;
                iBestInstance = i;
            }
        }
        return m_nClusterNr[iBestInstance];
    }
    
    //return the number of all clusters
    @Override
    public int numberOfClusters() {
        return Math.min(m_nNumClusters, m_instances.numInstances());
    }
    
    //setter
    public void setNumClusters(int nClusters) {
        m_nNumClusters = Math.max(1,nClusters);
    }
    
    public void setLinkType(int newLinkType) {
        m_nLinkType = newLinkType;
    }
    
    //getter
    public int getNumClusters() {
        return m_nNumClusters;
    }

    public int getLinkType() {
        return m_nLinkType;
    }
    
    //info about the algorithm and link type
    @Override
    public String toString() {
        StringBuilder temp = new StringBuilder();
        
        //output link type
        temp.append("\nmyAgnes: ");
        if (this.getLinkType()==SINGLE)
            temp.append("Single Link");
        else if (this.getLinkType()==COMPLETE)
            temp.append("Complete Link");
        temp.append("\n\n");
        
        //output instances in all cluster
        int clusterID = 0;
        for (int i=0;i<m_instances.numInstances();i++){
            if (!nClusterID[i].isEmpty()){
                temp.append("Cluster ").append(clusterID).append(": ");
                temp.append(nClusterID[i].toString());
                temp.append("\n");
                clusterID += 1;
            }
        }
        temp.append("\n");
        return temp.toString();
    }
}
