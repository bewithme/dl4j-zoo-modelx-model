package org.freeware.dl4j.modelx.model.human.face;

import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.CacheMode;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ActivationLayer;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.DropoutLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.zoo.ModelMetaData;
import org.deeplearning4j.zoo.PretrainedType;
import org.deeplearning4j.zoo.ZooModel;
import org.deeplearning4j.zoo.ZooType;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.learning.config.IUpdater;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import lombok.AllArgsConstructor;
import lombok.Builder;


/**
 * 
 * This model is used for human facial
 * 68 key points detection
 * 
 * @author wenfengxu wechatId:italybaby
 *
 */
@AllArgsConstructor
@Builder

public class Facial68KeypointsDetection extends ZooModel {

    @Builder.Default private long seed = 1234;
    @Builder.Default private int[] inputShape = new int[] {1, 96, 96};
    @Builder.Default private IUpdater updater = new Adam(0.001);
    @Builder.Default private CacheMode cacheMode = CacheMode.NONE;
    @Builder.Default private WorkspaceMode workspaceMode = WorkspaceMode.ENABLED;
    @Builder.Default private ConvolutionLayer.AlgoMode cudnnAlgoMode = ConvolutionLayer.AlgoMode.PREFER_FASTEST;
    @Builder.Default private int numClasses = 0;
    
 
    
    private Facial68KeypointsDetection() {}

    @Override
    public String pretrainedUrl(PretrainedType pretrainedType) {
        return null;
    }

    @Override
    public long pretrainedChecksum(PretrainedType pretrainedType) {
        return 0L;
    }

    @Override
    public Class<? extends Model> modelType() {
        return ComputationGraph.class;
    }

    @SuppressWarnings("unchecked")
	@Override
    public ComputationGraph init() {
      
    	String moduleName="facial-keypoints-detection";
        
        ComputationGraphConfiguration.GraphBuilder graph = graphBuilder("input",moduleName);
        
     
        graph.addInputs("input").setInputTypes(InputType.convolutional(inputShape[2], inputShape[1], inputShape[0]))
                         
                        .addLayer("outputLayer",
                                        new OutputLayer.Builder()
                                                        .lossFunction(LossFunctions.LossFunction.MSE)
                                                        .activation(Activation.IDENTITY)
                                                        .nOut(numClasses).build(),
                                                        createLayerName(moduleName,"dropout",18))
                        .setOutputs("outputLayer");
        
       
   
        ComputationGraphConfiguration conf = graph.build();
        
        ComputationGraph model = new ComputationGraph(conf);
        
        model.init();

        return model;
    }
    
    private String createLayerName(String moduleName, String leyerName,Integer blockIndex) {
	    String newLayerName=moduleName.concat("-").concat(leyerName).concat("-").concat(String.valueOf(blockIndex));
   	    return newLayerName;
    }
    
   
   private ComputationGraphConfiguration.GraphBuilder convBlock(ComputationGraphConfiguration.GraphBuilder graph,String moduleName,int blockIndex,String input,int[] kernelSize, int[] stride,int in,int out,ConvolutionMode convolutionMode) {
	   
	   ConvolutionLayer.Builder builder=new ConvolutionLayer.Builder(
    		   kernelSize, 
    		   stride);
       if(in>0){
    	   builder.nIn(in);
       }
	   
	   graph.addLayer(createLayerName(moduleName,"cnn",blockIndex),
   			            builder
                       .nOut(out)
                       .convolutionMode(convolutionMode)
                       .cudnnAlgoMode(cudnnAlgoMode)
                       .build(),
               input);
	   
	   graph.addLayer(createLayerName(moduleName,"activation-layer",blockIndex), 
					new ActivationLayer.Builder()
					.activation(Activation.RELU)
					.build(), createLayerName(moduleName,"cnn",blockIndex));
	   
	   return graph;
   }
   
   private ComputationGraphConfiguration.GraphBuilder convBlock(ComputationGraphConfiguration.GraphBuilder graph,String moduleName,int blockIndex,String input,int[] kernelSize, int[] stride,int out,ConvolutionMode convolutionMode) {
	   int in=0;
	   return convBlock(graph, moduleName, blockIndex, input, kernelSize, stride,in, out, convolutionMode);
   }
   
   private ComputationGraphConfiguration.GraphBuilder MaxPooling2D(ComputationGraphConfiguration.GraphBuilder graph, String moduleName,Integer blockIndex,String input,int[] kernelSize, int[] stride,ConvolutionMode convolutionMode) {
	   
	   graph
   
 	    .addLayer(createLayerName(moduleName,"max-pooling",blockIndex), 
				  new SubsamplingLayer.Builder(
				  SubsamplingLayer.PoolingType.MAX,
				  kernelSize, 
				  stride)
				  .convolutionMode(convolutionMode)
				  .build(),
				  input);
	   return graph;
   }
   
 
   private ComputationGraphConfiguration.GraphBuilder dropout(ComputationGraphConfiguration.GraphBuilder graph, String moduleName,Integer blockIndex,String input,double dropout) {

       graph.addLayer(createLayerName(moduleName,"dropout",blockIndex),
                  new DropoutLayer.Builder(dropout)
                  .build(),
                  input);
       return graph;
   }
    
  
   public ComputationGraphConfiguration.GraphBuilder graphBuilder(String input, String moduleName) {

        ComputationGraphConfiguration.GraphBuilder graph=new NeuralNetConfiguration.Builder().seed(seed)
                        .activation(Activation.RELU)
                        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                        .updater(updater)
                        .cacheMode(cacheMode)
                        .weightInit(WeightInit.XAVIER)
                        .trainingWorkspaceMode(workspaceMode)
                        .inferenceWorkspaceMode(workspaceMode)
                        .convolutionMode(ConvolutionMode.Same).graphBuilder();
        
       
        
     
        
        convBlock(graph, moduleName, 1, input, new int[] {5,5}, new int[] {1,1}, inputShape[0],32, ConvolutionMode.Same);
         
        MaxPooling2D(graph, moduleName, 2,  createLayerName(moduleName,"activation-layer",1),  new int[] {2,2},  new int[] {2,2}, ConvolutionMode.Same);
        
        dropout(graph, moduleName, 3, createLayerName(moduleName,"max-pooling",2), 0.1);
        
        
        
        convBlock(graph, moduleName, 4, createLayerName(moduleName,"dropout",3), new int[] {3,3}, new int[] {1,1}, 64, ConvolutionMode.Same);
        
        MaxPooling2D(graph, moduleName, 5, createLayerName(moduleName,"activation-layer",4),  new int[] {2,2},  new int[] {2,2}, ConvolutionMode.Same);
       
        dropout(graph, moduleName, 6, createLayerName(moduleName,"max-pooling",5), 0.2);
        
        convBlock(graph, moduleName, 7, createLayerName(moduleName,"dropout",6), new int[] {3,3}, new int[] {1,1}, 128, ConvolutionMode.Same);
        
        MaxPooling2D(graph, moduleName, 8,  createLayerName(moduleName,"activation-layer",7),  new int[] {2,2},  new int[] {2,2}, ConvolutionMode.Same);
        
        dropout(graph, moduleName, 9, createLayerName(moduleName,"max-pooling",8), 0.3);
        
        convBlock(graph, moduleName, 10, createLayerName(moduleName,"dropout",9), new int[] {2,2}, new int[] {1,1}, 256, ConvolutionMode.Same);
        
        MaxPooling2D(graph, moduleName,11,createLayerName(moduleName,"activation-layer",10),  new int[] {2,2},  new int[] {2,2}, ConvolutionMode.Same);
   
        dropout(graph, moduleName, 12, createLayerName(moduleName,"max-pooling",11), 0.4);
        
        graph.addLayer(createLayerName(moduleName,"dense-layer",13), new DenseLayer.Builder().nOut(1000).build(),createLayerName(moduleName,"dropout",12));
       
        graph.addLayer(createLayerName(moduleName,"activation-layer",14), new ActivationLayer.Builder().activation(Activation.RELU).build(), createLayerName(moduleName,"dense-layer",13));
        
        dropout(graph, moduleName, 15, createLayerName(moduleName,"activation-layer",14), 0.5);
         
        graph.addLayer(createLayerName(moduleName,"dense-layer",16), new DenseLayer.Builder().nOut(1000).build(),createLayerName(moduleName,"dropout",15));
        
        graph.addLayer(createLayerName(moduleName,"activation-layer",17), new ActivationLayer.Builder().activation(Activation.RELU).build(), createLayerName(moduleName,"dense-layer",16));
        
        dropout(graph, moduleName, 18, createLayerName(moduleName,"activation-layer",17), 0.6);
        
        return graph;


    }

    @Override
    public ModelMetaData metaData() {
        return new ModelMetaData(new int[][] {inputShape}, 1, ZooType.CNN);
    }

    @Override
    public void setInputShape(int[][] inputShape) {
        this.inputShape = inputShape[0];
    }

}
