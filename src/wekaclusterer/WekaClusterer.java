package wekaclusterer;

import java.util.Scanner;
import weka.clusterers.ClusterEvaluation;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;
import weka.filters.unsupervised.attribute.Remove;

public class WekaClusterer {

    public static void main(String[] args) throws Exception {
        Scanner sc = new Scanner(System.in);
        //load data
        String filedataset = "iris.arff"; //perlu input nama file harusnya
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
        System.out.print("Input: ");  
        int input = sc.nextInt(); sc.nextLine();
        
        //tentukan opsi algoritma yang dipiilih
        String[] options = new String[4];
        options[0] = "-N"; // number of cluster
        System.out.print("Tentukan jumlah cluster: "); //3 cluster untuk dataset iris
        options[1] = sc.nextInt() + "";
        options[2] = "-L"; // link type (SINGLE-LINK/COMPLETE-LINK)
        
        ClusterEvaluation eval = new ClusterEvaluation();
        if (input==1){ //build K-Means clusterer
            myKMeans clusterer = new myKMeans();   // new instance of clusterer
            clusterer.setNumClusters(Integer.parseInt(options[1]));
            clusterer.buildClusterer(dataClusterer);    // build the clusterer
            eval.setClusterer(clusterer); // the cluster to evaluate
        } else if (input==2) {
            myAgnes clusterer = new myAgnes();
            options[3] = "SINGLE";
            clusterer.setOptions(options);
            clusterer.buildClusterer(dataClusterer);    // build the clusterer
            eval.setClusterer(clusterer); // the cluster to evaluate
        } else if (input==3) {
            myAgnes clusterer = new myAgnes();
            options[3] = "COMPLETE";
            clusterer.setOptions(options);
            clusterer.buildClusterer(dataClusterer);    // build the clusterer
            eval.setClusterer(clusterer); // t8he cluster to evaluate
        }
        
        eval.evaluateClusterer(dataset); // data to evaluate the clusterer on
        System.out.println(eval.clusterResultsToString());  //output the result
    }
}
