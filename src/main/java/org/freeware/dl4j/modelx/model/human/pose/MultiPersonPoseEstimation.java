package org.freeware.dl4j.modelx.model.human.pose;

import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.CacheMode;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.conf.graph.MergeVertex;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ActivationLayer;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
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
import lombok.extern.slf4j.Slf4j;


/**
 * 
 * 
 * 
 * @author wenfengxu
 *
 */
@AllArgsConstructor
@Builder
@Slf4j
public class MultiPersonPoseEstimation extends ZooModel {

    @Builder.Default private long seed = 1234;
    @Builder.Default private int[] inputShape = new int[] {1, 96, 96};
    @Builder.Default private IUpdater updater = new Adam(0.0001, 0.9, 0.999,1e-08);
    @Builder.Default private CacheMode cacheMode = CacheMode.NONE;
    @Builder.Default private WorkspaceMode workspaceMode = WorkspaceMode.ENABLED;
    @Builder.Default private ConvolutionLayer.AlgoMode cudnnAlgoMode = ConvolutionLayer.AlgoMode.PREFER_FASTEST;

    
 
    
    private MultiPersonPoseEstimation() {}

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
      
    	String moduleName="vgg-block";
        
        ComputationGraphConfiguration.GraphBuilder graph = graphBuilder("input",moduleName);
        
     
        graph.addInputs("input").setInputTypes(InputType.convolutional(inputShape[2], inputShape[1], inputShape[0]))
                         
                        
                        .setOutputs("stage-t-left-cnn-6-6","stage-t-right-cnn-6-6");
        
       
        ComputationGraphConfiguration conf = graph.build();
        
        ComputationGraph model = new ComputationGraph(conf);
        
        model.init();

        return model;
    }
    
    
  
    
    
    private String createLayerName(String moduleName, String leyerName,Integer moduleIndex,Integer blockIndex) {
	    String newLayerName=moduleName.concat("-").concat(leyerName).concat("-").concat(String.valueOf(blockIndex)).concat("-").concat(String.valueOf(moduleIndex));
     	//log.info(newLayerName);
	    return newLayerName;
   }
   
   
    
   
   private ComputationGraphConfiguration.GraphBuilder convBlock(ComputationGraphConfiguration.GraphBuilder graph,String moduleName,Integer moduleIndex,int blockIndex,String input,int[] kernelSize, int[] stride,int in,int out,ConvolutionMode convolutionMode,double weightDecay,double biasDecay) {

    	cnn(graph, moduleName,moduleIndex, blockIndex,input, kernelSize,stride, in, out, convolutionMode, weightDecay, biasDecay);
	   
	    relu(graph, moduleName, moduleIndex, blockIndex);
	   
	   return graph;
   }

	public void cnn(ComputationGraphConfiguration.GraphBuilder graph, String moduleName,Integer moduleIndex,int blockIndex, String input,int[] kernelSize, int[] stride, int in, int out, ConvolutionMode convolutionMode, double weightDecay,double biasDecay) {
		    
		   ConvolutionLayer.Builder builder=new ConvolutionLayer.Builder(kernelSize, stride);
		   
		   if(in>0){
	    	   builder.nIn(in);
	       }
	       if(weightDecay>0) {
	    	   builder.l2(weightDecay);
	       }
	       if(biasDecay>0) {
	       	   builder.l2Bias(biasDecay); 
	       }
		   
		   graph.addLayer(createLayerName(moduleName,"cnn",moduleIndex,blockIndex),
	   			            builder
	                       .nOut(out)
	                       .biasInit(0.0)
	                       .weightInit(WeightInit.NORMAL)
	                       .convolutionMode(convolutionMode)
	                       .cudnnAlgoMode(cudnnAlgoMode)
	                       .build(),
	               input);
	}

	public void relu(ComputationGraphConfiguration.GraphBuilder graph, String moduleName, Integer moduleIndex,
			int blockIndex) {
		graph.addLayer(createLayerName(moduleName,"activation-layer",moduleIndex,blockIndex), 
						new ActivationLayer.Builder()
						.activation(Activation.RELU)
						.build(), createLayerName(moduleName,"cnn",moduleIndex,blockIndex));
	}
   
   private ComputationGraphConfiguration.GraphBuilder convBlock(ComputationGraphConfiguration.GraphBuilder graph,String moduleName,int moduleIndex,int blockIndex,String input,int[] kernelSize, int out,double weightDecay,double biasDecay) {
	   int in=0;
	   int[] stride=new int[] {1,1};
	   return convBlock(graph, moduleName,moduleIndex, blockIndex, input, kernelSize, stride,in, out, ConvolutionMode.Same, weightDecay, biasDecay);
   }
   
   private ComputationGraphConfiguration.GraphBuilder convBlock(ComputationGraphConfiguration.GraphBuilder graph,String moduleName,int moduleIndex,int blockIndex,String input,int[] kernelSize,int in, int out,double weightDecay,double biasDecay) {
	   int[] stride=new int[] {1,1};
	   return convBlock(graph, moduleName,moduleIndex, blockIndex, input, kernelSize, stride,in, out, ConvolutionMode.Same, weightDecay, biasDecay);
   }
   
 
   
   private ComputationGraphConfiguration.GraphBuilder vggBlock(ComputationGraphConfiguration.GraphBuilder graph,String moduleName,int moduleIndex,String input,int in,double weightDecay) {
	  
	   int[] kernelSize=new  int[]{3,3};
	   
	   //block 1
	   convBlock(graph, moduleName,moduleIndex, 0, input, kernelSize, in,64, weightDecay, 0);
	   
	   convBlock(graph, moduleName, moduleIndex,1, createLayerName(moduleName,"activation-layer",moduleIndex,0), kernelSize,64, weightDecay, 0);
	   
	   MaxPooling2D(graph,moduleName,moduleIndex, 2, createLayerName(moduleName,"activation-layer",moduleIndex,1), new int[] {2,2},new int[] {2,2});
	   
	   
	   //block 2
       convBlock(graph, moduleName,moduleIndex, 3, createLayerName(moduleName,"max-pooling",moduleIndex,2), kernelSize,128, weightDecay, 0);
	   
	   convBlock(graph, moduleName,moduleIndex, 4, createLayerName(moduleName,"activation-layer",moduleIndex,3), kernelSize,128, weightDecay, 0);
	   
	   MaxPooling2D(graph,moduleName,moduleIndex, 5, createLayerName(moduleName,"activation-layer",moduleIndex,4), new int[] {2,2},new int[] {2,2});
	   
	  
	   //block 3
	   convBlock(graph, moduleName, moduleIndex,6, createLayerName(moduleName,"max-pooling",moduleIndex,5), kernelSize,256, weightDecay, 0);
	   
	   convBlock(graph, moduleName,moduleIndex, 7, createLayerName(moduleName,"activation-layer",moduleIndex,6), kernelSize,256, weightDecay, 0);
	   
	   convBlock(graph, moduleName,moduleIndex, 8, createLayerName(moduleName,"activation-layer",moduleIndex,7), kernelSize,256, weightDecay, 0);
	   
	   convBlock(graph, moduleName,moduleIndex, 9, createLayerName(moduleName,"activation-layer",moduleIndex,8), kernelSize,256, weightDecay, 0);
	   
	   MaxPooling2D(graph, moduleName,moduleIndex, 10, createLayerName(moduleName,"activation-layer",moduleIndex,9), new int[] {2,2},new int[] {2,2});
	   
	   //block 4
	   convBlock(graph, moduleName,moduleIndex, 11, createLayerName(moduleName,"max-pooling",moduleIndex,10), kernelSize,512, weightDecay, 0);
	   
	   convBlock(graph, moduleName,moduleIndex, 12, createLayerName(moduleName,"activation-layer",moduleIndex,11), kernelSize,512, weightDecay, 0);
	   
	   
	   //Additional non vgg layers
	   convBlock(graph, moduleName,moduleIndex, 13, createLayerName(moduleName,"activation-layer",moduleIndex,12), kernelSize,256, weightDecay, 0);
	   
	   convBlock(graph, moduleName,moduleIndex, 14, createLayerName(moduleName,"activation-layer",moduleIndex,13), kernelSize,128, weightDecay, 0);

	   return graph;
	   
   }
   
   private ComputationGraphConfiguration.GraphBuilder stage1Block(ComputationGraphConfiguration.GraphBuilder graph,String moduleName,int moduleIndex,String input,int out,double weightDecay) {
		  
	   int[] kernelSize=new  int[]{3,3};
	   
	   convBlock(graph, moduleName,moduleIndex, 0, input, kernelSize, 128, weightDecay, 0);
	   
	   convBlock(graph, moduleName,moduleIndex, 1, createLayerName(moduleName,"activation-layer",moduleIndex,0), kernelSize,128, weightDecay, 0);
	   
	   convBlock(graph, moduleName,moduleIndex, 2, createLayerName(moduleName,"activation-layer",moduleIndex,1), kernelSize,128, weightDecay, 0);
		
	   convBlock(graph, moduleName,moduleIndex, 3, createLayerName(moduleName,"activation-layer",moduleIndex,2), new  int[]{1,1},512, weightDecay, 0);

	   cnn(graph, moduleName, moduleIndex,4,createLayerName(moduleName,"activation-layer",moduleIndex,3), new int[]{1,1},new int[]{1,1},0,out, ConvolutionMode.Same, weightDecay, 0);
	   
	   return graph;
	   
   }
   
   private ComputationGraphConfiguration.GraphBuilder stageTBlock(ComputationGraphConfiguration.GraphBuilder graph,String moduleName,int moduleIndex,String input,int out,double weightDecay) {
		  
	   int[] kernelSize=new  int[]{7,7};
	   
	   convBlock(graph, moduleName,moduleIndex, 0, input, kernelSize, 128, weightDecay, 0);
	   
	   convBlock(graph, moduleName,moduleIndex, 1, createLayerName(moduleName,"activation-layer",moduleIndex,0), kernelSize,128, weightDecay, 0);
	   
	   convBlock(graph, moduleName,moduleIndex, 2, createLayerName(moduleName,"activation-layer",moduleIndex,1), kernelSize,128, weightDecay, 0);
		
	   convBlock(graph, moduleName,moduleIndex, 3, createLayerName(moduleName,"activation-layer",moduleIndex,2), kernelSize,128, weightDecay, 0);

	   convBlock(graph, moduleName,moduleIndex, 4, createLayerName(moduleName,"activation-layer",moduleIndex,3), kernelSize,128, weightDecay, 0);
	   
	   convBlock(graph, moduleName,moduleIndex, 5, createLayerName(moduleName,"activation-layer",moduleIndex,4), new  int[]{1,1},128, weightDecay, 0);
	   
	   cnn(graph, moduleName, moduleIndex, 6, createLayerName(moduleName,"activation-layer",moduleIndex,5), new int[]{1,1},new int[]{1,1},0,out, ConvolutionMode.Same, weightDecay, 0);
	   
	   return graph;
	   
   }
   
   
   
  
   
 
 private ComputationGraphConfiguration.GraphBuilder MaxPooling2D(ComputationGraphConfiguration.GraphBuilder graph, String moduleName,int moduleIndex,Integer blockIndex,String input,int[] kernelSize, int[] stride) {
	   
	   graph
   
 	    .addLayer(createLayerName(moduleName,"max-pooling",moduleIndex,blockIndex), 
				  new SubsamplingLayer.Builder(
				  SubsamplingLayer.PoolingType.MAX,
				  kernelSize, 
				  stride)
				  .convolutionMode(ConvolutionMode.Same)
				  .build(),
				  input);
	   return graph;
   }
 

    public ComputationGraphConfiguration.GraphBuilder graphBuilder(String input, String moduleName) {

        ComputationGraphConfiguration.GraphBuilder graph = new NeuralNetConfiguration.Builder().seed(seed)
                        .activation(Activation.RELU)
                        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                        .updater(updater)
                        .cacheMode(cacheMode)
                        .trainingWorkspaceMode(workspaceMode)
                        .inferenceWorkspaceMode(workspaceMode)
                        .convolutionMode(ConvolutionMode.Same).graphBuilder();
        
       int weightDecay=0;
       
       int vggBlockModuleIndex=0; 
       
       vggBlock(graph, moduleName,vggBlockModuleIndex,input, inputShape[0], weightDecay);
       
       String stage1ModuleName="stage1-block";
       
       String vggBlockOutput=createLayerName(moduleName,"activation-layer",vggBlockModuleIndex,14);
       
       String stage1Left=stage1ModuleName.concat("-left");
       
       String stage1Right=stage1ModuleName.concat("-right");
       
       stage1Block(graph, stage1Left, 0, vggBlockOutput, 38, weightDecay);
       
       stage1Block(graph, stage1Right, 0, vggBlockOutput, 19, weightDecay);
       
       String stage1BlockLeftOutput=createLayerName(stage1Left, "cnn", 0, 4);
       
       String stage1BlockRightOutput=createLayerName(stage1Right, "cnn", 0, 4);
       
       
       graph.addVertex(
    		     createLayerName(stage1ModuleName,"merge-vertex",0,0), 
			     new MergeVertex(),
			     new String[]{vggBlockOutput,stage1BlockLeftOutput,stage1BlockRightOutput});
       
       int stages=6;
       
       String stageTModuleName="stage-t";
       
       String stageTInputName=createLayerName(stage1ModuleName,"merge-vertex",0,0);
       
       
       for(int sn=2;sn<stages+1;sn++) {
    	   
    	   if(sn>2) {
    		   stageTInputName=createLayerName(stageTModuleName,"merge-vertex",sn-1,0);
    	   }
    	   
    	   String stageLeftName=stageTModuleName.concat("-left");
    	   
    	   stageTBlock(graph, stageLeftName, sn,  stageTInputName, 38, weightDecay);
    	   
    	   String stageLeftOutput=createLayerName(stageLeftName, "cnn", sn, 6);
    	   
    	   String stageRightName=stageTModuleName.concat("-right");
    	   
    	   stageTBlock(graph, stageRightName, sn,stageTInputName, 19, weightDecay);
    	   
    	   String stageRightOutput=createLayerName(stageRightName, "cnn", sn, 6);
    	   
    	   if(sn<stages) {
    		   graph.addVertex(
    	    		     createLayerName(stageTModuleName,"merge-vertex",sn,0), 
    				     new MergeVertex(),
    				     new String[]{vggBlockOutput,stageLeftOutput,stageRightOutput});
    	   }
    	   
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
