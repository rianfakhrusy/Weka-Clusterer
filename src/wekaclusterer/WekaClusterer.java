package wekaclusterer;

import java.util.Scanner;
import weka.clusterers.ClusterEvaluation;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;
import weka.filters.unsupervised.attribute.Remove;

public class WekaClusterer {
    private static boolean KELUAR = false;
    private static String[] options = new String[4];
    private static int cluster;
    
    public static void main(String[] args) throws Exception {
        while(!KELUAR){
            Scanner sc = new Scanner(System.in);
            //System.out.println("Masukkan nama data set : ");
            String filedataset = "cri.arff";
            Instances dataset = new ConverterUtils.DataSource(filedataset).getDataSet();
            if(dataset.classIndex() == -1)
                dataset.setClassIndex(dataset.numAttributes() - 1);
            dataset.deleteStringAttributes();

            //remove class attribute
            Instances dataClusterer = null;
            Remove filter = new Remove();
            filter.setAttributeIndices("" + (dataset.classIndex() + 1));
            try {
                filter.setInputFormat(dataset);
                dataClusterer = filter.useFilter(dataset, filter);
            } catch (Exception e1) {
                e1.printStackTrace();
                return;
            }

            //pilih algoritma
            System.out.println("Masukkan algoritma clustering");
            System.out.println("1. Simple K-Means");
            System.out.println("2. Agglomerative Clustering : Single Link");
            System.out.println("3. Agglomerative Clustering : Complete Link");
            System.out.println("4. Keluar");
            System.out.print("Input: ");  
            int input = sc.nextInt(); sc.nextLine();   
            
            ClusterEvaluation eval = new ClusterEvaluation();
            switch(input){
                case 1 : {
                            System.out.print("Tentukan jumlah cluster: ");
                            cluster = sc.nextInt();
                            myKMeans clusterer = new myKMeans();   // new instance of clusterer
                            clusterer.setNumClusters(cluster);
                            clusterer.buildClusterer(dataClusterer);    // build the clusterer
                            eval.setClusterer(clusterer); // the cluster to evaluate
                            eval.evaluateClusterer(dataset); // data to evaluate the clusterer on
                            System.out.println(eval.clusterResultsToString());  //output the result
                            break;
                        }
                case 2 : {
                            System.out.print("Tentukan jumlah cluster: ");
                            cluster = sc.nextInt();
                            myAgnes clusterer = new myAgnes();
                            clusterer.setNumClusters(cluster);
                            clusterer.setLinkType(0);
                            clusterer.buildClusterer(dataClusterer);    // build the clusterer
                            eval.setClusterer(clusterer); // the cluster to evaluate
                            eval.evaluateClusterer(dataset); // data to evaluate the clusterer on
                            System.out.println(eval.clusterResultsToString());  //output the result
                            break;
                        }
                case 3 : {
                            System.out.print("Tentukan jumlah cluster: ");
                            cluster = sc.nextInt();
                            myAgnes clusterer = new myAgnes();
                            clusterer.setNumClusters(cluster);
                            clusterer.setLinkType(1);
                            clusterer.buildClusterer(dataClusterer);    // build the clusterer
                            eval.setClusterer(clusterer); // the cluster to evaluate
                            eval.evaluateClusterer(dataset); // data to evaluate the clusterer on
                            System.out.println(eval.clusterResultsToString());  //output the result
                            break;
                        }
                case 4 : {
                            KELUAR = true;
                            break;
                        }
                default :{
                            System.out.println("Anda memberi input yang salah !");
                            break;
                        }
            }
        }
    }
}
