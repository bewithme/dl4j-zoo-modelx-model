package org.freeware.dl4j.modelx.model.image;

import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.CacheMode;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.conf.distribution.TruncatedNormalDistribution;
import org.deeplearning4j.nn.conf.graph.L2NormalizeVertex;
import org.deeplearning4j.nn.conf.graph.MergeVertex;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.zoo.ModelMetaData;
import org.deeplearning4j.zoo.PretrainedType;
import org.deeplearning4j.zoo.ZooModel;
import org.deeplearning4j.zoo.ZooType;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.IUpdater;
import org.nd4j.linalg.learning.config.RmsProp;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.extern.slf4j.Slf4j;


/**
 * 
 * https://arxiv.org/pdf/1602.07261v1.pdf
 * @author wenfengxu
 *
 */
@AllArgsConstructor
@Builder
@Slf4j
public class InceptionV4 extends ZooModel {

    @Builder.Default private long seed = 1234;
    @Builder.Default private int[] inputShape = new int[] {3, 299, 299};
    @Builder.Default private int numClasses = 0;
    @Builder.Default private IUpdater updater = new RmsProp(0.1, 0.96, 0.001);
    @Builder.Default private CacheMode cacheMode = CacheMode.NONE;
    @Builder.Default private WorkspaceMode workspaceMode = WorkspaceMode.ENABLED;
    @Builder.Default private ConvolutionLayer.AlgoMode cudnnAlgoMode = ConvolutionLayer.AlgoMode.PREFER_FASTEST;

    
   private static String activationLayerName="activation-layer";
    
   private static String mergeVertexLayerName="merge-vertex";
    
   private static String maxPoolingLayerName="max-pooling";
    
    private InceptionV4() {}

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
      
    	int embeddingSize = 128;
        
        ComputationGraphConfiguration.GraphBuilder graph = graphBuilder("input");
        
        String inceptionAInput=createLayerName("stem",mergeVertexLayerName,0,16);

        graph.addInputs("input").setInputTypes(InputType.convolutional(inputShape[2], inputShape[1], inputShape[0]))





				.addLayer("outputLayer",new OutputLayer.Builder().nOut(numClasses)
								.lossFunction(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
								.activation(Activation.SOFTMAX).build()

						,
						inceptionAInput)
                        .setOutputs("outputLayer");

        ComputationGraphConfiguration conf = graph.build();
        ComputationGraph model = new ComputationGraph(conf);
        model.init();

        return model;
    }
    
    
  
    
    
   private String createLayerName(String moduleName, String leyerName,Integer moduleIndex,Integer blockIndex) {
	    String newLayerName=moduleName.concat("-").concat(leyerName).concat("-").concat(String.valueOf(moduleIndex)).concat("-").concat(String.valueOf(blockIndex));
     	return newLayerName;
   }
   

    
   
   private ComputationGraphConfiguration.GraphBuilder convBlock(ComputationGraphConfiguration.GraphBuilder graph,String moduleName,int moduleIndex,int blockIndex,String input,int[] kernelSize, int[] stride,int in,int out,ConvolutionMode convolutionMode) {
	   
	   ConvolutionLayer.Builder builder=new ConvolutionLayer.Builder(
    		   kernelSize, 
    		   stride);
       if(in>0){
    	   builder.nIn(in);
       }
	   
	   graph.addLayer(createLayerName(moduleName,"cnn",moduleIndex,blockIndex),
   			            builder
                       .nOut(out)
                       .convolutionMode(convolutionMode)
                       .cudnnAlgoMode(cudnnAlgoMode)
                       .build(),
               input);

	   batchNormAndActivation(graph, moduleName,moduleIndex, blockIndex, out);

	   return graph;
   }

	private void batchNormAndActivation(ComputationGraphConfiguration.GraphBuilder graph, String moduleName,int moduleIndex,int blockIndex, int out) {
		graph.addLayer(createLayerName(moduleName,"batch",moduleIndex,blockIndex),
					new BatchNormalization.Builder(false)
				   .decay(0.99)
				   .eps(0.001)
				   .nOut(out)
				   .build(),
				   createLayerName(moduleName,"cnn",moduleIndex,blockIndex));
		graph.addLayer(createLayerName(moduleName,"activation-layer",moduleIndex,blockIndex),
					 new ActivationLayer.Builder()
					 .activation(Activation.RELU)
					 .build(), createLayerName(moduleName,"batch",moduleIndex,blockIndex));
	}

	private ComputationGraphConfiguration.GraphBuilder convBlock(ComputationGraphConfiguration.GraphBuilder graph,String moduleName,int moduleIndex,int blockIndex,String input,int[] kernelSize, int[] stride,int out,ConvolutionMode convolutionMode) {
	   int in=0;
	   return convBlock(graph, moduleName, moduleIndex,blockIndex, input, kernelSize, stride,in, out, convolutionMode);
   }
   
   private ComputationGraphConfiguration.GraphBuilder convBlock(ComputationGraphConfiguration.GraphBuilder graph,String moduleName,int moduleIndex,int blockIndex,String input,int[] kernelSize,int out,ConvolutionMode convolutionMode) {
	   int in=0;
	   int[] stride= {1,1};
	   return convBlock(graph, moduleName,moduleIndex, blockIndex, input, kernelSize, stride,in, out, convolutionMode);
   }
   private ComputationGraphConfiguration.GraphBuilder convBlock(ComputationGraphConfiguration.GraphBuilder graph,String moduleName,int moduleIndex,int blockIndex,String input,int[] kernelSize,int out) {
	   int in=0;
	   int[] stride= {1,1};
	   ConvolutionMode convolutionMode=ConvolutionMode.Same;
	   return convBlock(graph, moduleName,moduleIndex, blockIndex, input, kernelSize, stride,in, out, convolutionMode);
   }
    
   private ComputationGraphConfiguration.GraphBuilder MaxPooling2D(ComputationGraphConfiguration.GraphBuilder graph,int[] kernelSize, int[] stride,ConvolutionMode convolutionMode, String moduleName, Integer moduleIndex, Integer blockIndex,String input) {
	   
	   graph
   
 	    .addLayer(createLayerName(moduleName,"max-pooling",moduleIndex,blockIndex),
				  new SubsamplingLayer.Builder(
				  SubsamplingLayer.PoolingType.MAX,
				  kernelSize, 
				  stride)
				  .convolutionMode(convolutionMode)
				  .build(),
				  input);
	   return graph;
   }
   
   private ComputationGraphConfiguration.GraphBuilder AveragePooling2D(ComputationGraphConfiguration.GraphBuilder graph,int[] kernelSize, int[] stride,ConvolutionMode convolutionMode, String moduleName, Integer moduleIndex, Integer blockIndex,String input) {
	   
	   graph
   
 	    .addLayer(createLayerName(moduleName,"avg-pooling",moduleIndex,blockIndex),
				  new SubsamplingLayer.Builder(
				  SubsamplingLayer.PoolingType.AVG,
				  kernelSize, 
				  stride)
				  .convolutionMode(convolutionMode)
				  .build(),
				  input);
	   return graph;
   }
   
 private ComputationGraphConfiguration.GraphBuilder MaxPooling2D(ComputationGraphConfiguration.GraphBuilder graph, String moduleName,Integer moduleIndex,Integer blockIndex,String input,int[] kernelSize, int[] stride,ConvolutionMode convolutionMode) {
	   
	   graph
   
 	    .addLayer(createLayerName(moduleName,"max-pooling",moduleIndex,blockIndex),
				  new SubsamplingLayer.Builder(
				  SubsamplingLayer.PoolingType.MAX,
				  kernelSize, 
				  stride)
				  .convolutionMode(convolutionMode)
				  .build(),
				  input);
	   return graph;
   }
   
   private ComputationGraphConfiguration.GraphBuilder AveragePooling2D(ComputationGraphConfiguration.GraphBuilder graph, String moduleName, Integer moduleIndex,Integer blockIndex,String input,int[] kernelSize, int[] stride,ConvolutionMode convolutionMode) {
	   
	   graph
   
 	    .addLayer(createLayerName(moduleName,"avg-pooling",moduleIndex,blockIndex),
				  new SubsamplingLayer.Builder(
				  SubsamplingLayer.PoolingType.AVG,
				  kernelSize, 
				  stride)
				  .convolutionMode(convolutionMode)
				  .build(),
				  input);
	   return graph;
   }
    
    
    /**
     * build InceptionA
     * @param graph
     * @param input
     * @param moduleIndex
     * @return
     */
   private ComputationGraphConfiguration.GraphBuilder buildInceptionA(ComputationGraphConfiguration.GraphBuilder graph,String input,Integer moduleIndex) {
    	
    	String moduleName="inception-A";


    //	batchNormAndActivation();
	  // convBlock(graph, moduleName, 1, input, new int[] {3,3},new int[] {2,2}, 32, ConvolutionMode.Truncate);


	   return graph;
    	
    }
    
    
   private ComputationGraphConfiguration.GraphBuilder buildReductionA(ComputationGraphConfiguration.GraphBuilder graph,String input,Integer moduleIndex) {
    	
    	String moduleName="reduction-A";




	   return null;
    }
    
 
   private ComputationGraphConfiguration.GraphBuilder buildInceptionC(ComputationGraphConfiguration.GraphBuilder graph,String input,Integer moduleIndex) {
    	
    	String moduleName="inception-C";

       	return null;
    	
    }
    
    
   private ComputationGraphConfiguration.GraphBuilder buildReductionB(ComputationGraphConfiguration.GraphBuilder graph,String input,Integer moduleIndex) {
    	
    	String moduleName="reduction-B";
    	

    return null;
    }
    
    
    
    /**
     * build stem for model
     * 
     * @param graph
     * @param input
     * @return
     */
   private ComputationGraphConfiguration.GraphBuilder buildInceptionStem(ComputationGraphConfiguration.GraphBuilder graph,String input) {
    	
	      String moduleName="stem";

	      int moduleIndex=0;

	      convBlock(graph, moduleName, moduleIndex,1, input, new int[] {3,3},new int[] {2,2}, inputShape[0], 32, ConvolutionMode.Truncate);
	   
	      convBlock(graph, moduleName,moduleIndex, 2, createLayerName(moduleName,activationLayerName,moduleIndex,1), new int[] {3,3}, 32, ConvolutionMode.Truncate);
	      
	      convBlock(graph, moduleName, moduleIndex,3, createLayerName(moduleName,activationLayerName,moduleIndex,2), new int[] {3,3}, 64);
    	 
	      MaxPooling2D(graph,moduleName,moduleIndex,4, createLayerName(moduleName,activationLayerName,moduleIndex,3), new int[] {3, 3}, new int[] {2, 2}, ConvolutionMode.Truncate);
	      
	      convBlock(graph, moduleName, moduleIndex,5, createLayerName(moduleName,activationLayerName,moduleIndex,3), new int[] {3,3},new int[] {2,2},  96, ConvolutionMode.Truncate);
		   
	      graph.addVertex(createLayerName(moduleName,mergeVertexLayerName,moduleIndex,6), new MergeVertex(), new String[]{createLayerName(moduleName,maxPoolingLayerName,moduleIndex,4),createLayerName(moduleName,activationLayerName,moduleIndex,5)});
	  
    	  //stem right branch start
	      convBlock(graph, moduleName,moduleIndex, 7, createLayerName(moduleName,mergeVertexLayerName,moduleIndex,6), new int[] {1,1}, 64);
	      
	      convBlock(graph, moduleName, moduleIndex,8, createLayerName(moduleName,activationLayerName,moduleIndex,7), new int[] {3,3}, 96, ConvolutionMode.Truncate);
		  
	      //stem right branch end
	      
	      //stem left start
	      convBlock(graph, moduleName, moduleIndex,9, createLayerName(moduleName,mergeVertexLayerName,moduleIndex,6), new int[] {1,1}, 64);
	      convBlock(graph, moduleName,moduleIndex, 10, createLayerName(moduleName,activationLayerName,moduleIndex,9), new int[] {1,7}, 64);
	      convBlock(graph, moduleName, moduleIndex,11, createLayerName(moduleName,activationLayerName,moduleIndex,10), new int[] {7,1}, 64);
	      convBlock(graph, moduleName, moduleIndex,12, createLayerName(moduleName,activationLayerName,moduleIndex,11), new int[] {3,3}, 96, ConvolutionMode.Truncate);
		  //stem left end
	      
	      graph.addVertex(createLayerName(moduleName,mergeVertexLayerName,moduleIndex,13), new MergeVertex(), new String[]{createLayerName(moduleName,activationLayerName,moduleIndex,12),createLayerName(moduleName,activationLayerName,moduleIndex,8)});
    	  
    	  convBlock(graph, moduleName, moduleIndex,14, createLayerName(moduleName,mergeVertexLayerName,moduleIndex,13), new int[] {3,3},new int[] {2,2},  192, ConvolutionMode.Truncate);
		  
	      MaxPooling2D(graph,moduleName,moduleIndex,15, createLayerName(moduleName,mergeVertexLayerName,moduleIndex,13), new int[] {3, 3}, new int[] {2, 2}, ConvolutionMode.Truncate);
		     
          graph.addVertex(createLayerName(moduleName,mergeVertexLayerName,moduleIndex,16), new MergeVertex(), new String[]{createLayerName(moduleName,maxPoolingLayerName,moduleIndex,15),createLayerName(moduleName,activationLayerName,moduleIndex,14)});
    	  
       	return graph;
    }

    public ComputationGraphConfiguration.GraphBuilder graphBuilder(String input) {

        ComputationGraphConfiguration.GraphBuilder graph = new NeuralNetConfiguration.Builder().seed(seed)
                        .activation(Activation.RELU)
                        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                        .updater(updater)
                        .weightInit(new TruncatedNormalDistribution(0.0, 0.5))
                        .l2(5e-5)
                        .miniBatch(true)
                        .cacheMode(cacheMode)
                        .trainingWorkspaceMode(workspaceMode)
                        .inferenceWorkspaceMode(workspaceMode)
                        .convolutionMode(ConvolutionMode.Truncate).graphBuilder();


       graph=buildInceptionStem(graph, input);
      
       
       //inceptionA input is the stem output
      /* String inceptionAInput=createLayerName("stem",mergeVertexLayerName,16); 
       
       String inceptionAModuleName="inception-A";
        
       int inceptionABatchSize=4;
       
       graph = buildBatchInceptionA(graph, inceptionAInput, inceptionAModuleName, inceptionABatchSize);
        
       String reductionAInput=createLayerName(inceptionAModuleName,"merge-vertex-0",inceptionABatchSize-1); 
       
       graph=buildReductionA(graph, reductionAInput, 0) ;
       
       
       //inceptionB input is the reductionA output
       String inceptionBInput=createLayerName("reduction-A","merge-vertex-0",0); 
       
       String inceptionBModuleName="inception-B";
        
       int inceptionBBatchSize=7;
       
       graph=buildBatchInceptionB(graph, inceptionBInput, inceptionBModuleName, inceptionBBatchSize);
       
       String reductionBInput=createLayerName(inceptionBModuleName,"merge-vertex-0",inceptionBBatchSize-1); 
    	
       graph=buildReductionB(graph, reductionBInput, 0) ;
       
       

       //inceptionC input is the reductionA output
       String inceptionCInput=createLayerName("reduction-B","merge-vertex-0",0); 
       
       String inceptionCModuleName="inception-C";
        
       int inceptionCBatchSize=3;
       
       graph=buildBatchInceptionC(graph, inceptionCInput, inceptionCModuleName, inceptionCBatchSize);
       */
    		   
       return graph;
    }

	private ComputationGraphConfiguration.GraphBuilder buildBatchInceptionA(ComputationGraphConfiguration.GraphBuilder graph,String input, String moduleName, int batchSize) {
		
		for(int i=0;i<batchSize;i++) {
		    
		       	if(i>0) {
		           input=createLayerName(moduleName,"merge-vertex-0",0,i-1);
		       	}
   
		       	graph=buildInceptionA(graph, input, i);
		    	
		   }
		return graph;
	}
	
	private ComputationGraphConfiguration.GraphBuilder buildBatchInceptionB(ComputationGraphConfiguration.GraphBuilder graph,String input, String moduleName, int batchSize) {
		
		for(int i=0;i<batchSize;i++) {
		    
		       	if(i>0) {
		           input=createLayerName(moduleName,"merge-vertex-0",0,i-1);
		       	}
   
		      // 	graph=buildInceptionB(graph, input, i);
		    	
		   }
		return graph;
	}
	
   private ComputationGraphConfiguration.GraphBuilder buildBatchInceptionC(ComputationGraphConfiguration.GraphBuilder graph,String input, String moduleName, int batchSize) {
		
		for(int i=0;i<batchSize;i++) {
		    
		       	if(i>0) {
		           input=createLayerName(moduleName,"merge-vertex-0",0,i-1);
		       	}
   
		       	graph=buildInceptionC(graph, input, i);
		    	
		 }
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
