package org.freeware.dl4j.modelx.model.image;

import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.CacheMode;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.conf.distribution.TruncatedNormalDistribution;
import org.deeplearning4j.nn.conf.graph.MergeVertex;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.zoo.ModelMetaData;
import org.deeplearning4j.zoo.PretrainedType;
import org.deeplearning4j.zoo.ZooModel;
import org.deeplearning4j.zoo.ZooType;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.learning.config.IUpdater;
import org.nd4j.linalg.learning.config.RmsProp;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.extern.slf4j.Slf4j;


/**
 * This is dl4j implement of InceptionV4
 * https://arxiv.org/pdf/1602.07261v1.pdf
 * @author wenfengxu  wechatid:italybaby
 *
 */
@AllArgsConstructor
@Builder
@Slf4j
public class InceptionV4 extends ZooModel {

    @Builder.Default private long seed = 1234;
    @Builder.Default private int[] inputShape = new int[] {3, 299, 299};
    @Builder.Default private int numClasses = 0;
    @Builder.Default private IUpdater updater = new Adam(0.001);
    @Builder.Default private CacheMode cacheMode = CacheMode.NONE;
    @Builder.Default private WorkspaceMode workspaceMode = WorkspaceMode.ENABLED;
    @Builder.Default private ConvolutionLayer.AlgoMode cudnnAlgoMode = ConvolutionLayer.AlgoMode.PREFER_FASTEST;

    
   private static String ACTIVATION_LAYER ="activation-layer";
    
   private static String MERGE_VERTEX ="merge-vertex";
    
   private static String MAX_POOLING ="max-pooling";

   private static String AVG_POOLING ="avg-pooling";
   
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

        
        ComputationGraphConfiguration.GraphBuilder graph = graphBuilder("input");


        graph.addInputs("input").setInputTypes(InputType.convolutional(inputShape[2], inputShape[1], inputShape[0]))

				.addLayer("outputLayer",new OutputLayer.Builder().nOut(numClasses)
								.lossFunction(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
								.activation(Activation.SOFTMAX)
								.build()
						,
						"drop_out_layer")
                        .setOutputs("outputLayer");

        ComputationGraphConfiguration conf = graph.build();

        ComputationGraph model = new ComputationGraph(conf);

        model.init();

        return model;
    }


	public ComputationGraphConfiguration.GraphBuilder graphBuilder(String input) {

		ComputationGraphConfiguration.GraphBuilder graph = new NeuralNetConfiguration.Builder().seed(seed)
				.activation(Activation.RELU)
				.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
				.updater(updater)
				.weightInit(WeightInit.XAVIER)
				.miniBatch(true)
				.cacheMode(cacheMode)
				.trainingWorkspaceMode(workspaceMode)
				.inferenceWorkspaceMode(workspaceMode)
				.convolutionMode(ConvolutionMode.Truncate)
				.graphBuilder();


		graph=buildInceptionStem(graph, input);

		String inceptionAInput=createLayerName("stem", MERGE_VERTEX,0,16);
		int inceptionABatchSize=4;
		graph = buildBatchInceptionA(graph, inceptionAInput, inceptionABatchSize);

        String reductionAInput=createLayerName("inception-A", MERGE_VERTEX,3,9);
        graph=buildReductionA(graph, reductionAInput, 0) ;


		String inceptionBInput=createLayerName("reduction-A", MERGE_VERTEX,0,4);
		int inceptionBBatchSize=7;
		graph =buildBatchInceptionB(graph, inceptionBInput,  inceptionBBatchSize);

		String reductionBInput=createLayerName("inception-B", MERGE_VERTEX,6,11);
		graph=buildReductionB(graph, reductionBInput, 0) ;


		String inceptionCInput=createLayerName("reduction-B", MERGE_VERTEX,0,7);
		int inceptionCBatchSize=3;
		graph =buildBatchInceptionC(graph, inceptionCInput,  inceptionCBatchSize);

		String maxPoolingInput=createLayerName("inception-C", MERGE_VERTEX,2,13);

		MaxPooling2D(graph,"max-pooling",0,0, maxPoolingInput, new int[] {8, 8}, new int[] {8, 8}, ConvolutionMode.Truncate);

        graph.addLayer("drop_out_layer",new DropoutLayer.Builder(0.8)
				.build(),createLayerName("max-pooling", MAX_POOLING,0,0));

		return graph;
	}

	private ComputationGraphConfiguration.GraphBuilder buildBatchInceptionA(ComputationGraphConfiguration.GraphBuilder graph,String input,int batchSize) {

		for(int i=0;i<batchSize;i++) {

			if(i>0) {
				input=createLayerName("inception-A", MERGE_VERTEX,i-1,9);
			}

			graph=buildInceptionA(graph, input, i);

		}
		return graph;
	}

	private ComputationGraphConfiguration.GraphBuilder buildBatchInceptionB(ComputationGraphConfiguration.GraphBuilder graph,String input,  int batchSize) {

		for(int i=0;i<batchSize;i++) {

			if(i>0) {
				input=createLayerName("inception-B", MERGE_VERTEX,i-1,11);
			}

			 graph=buildInceptionB(graph, input, i);

		}
		return graph;
	}

	private ComputationGraphConfiguration.GraphBuilder buildBatchInceptionC(ComputationGraphConfiguration.GraphBuilder graph,String input, int batchSize) {

		for(int i=0;i<batchSize;i++) {

			if(i>0) {
				input=createLayerName("inception-C",MERGE_VERTEX,i-1,13);
			}

			graph=buildInceptionC(graph, input, i);

		}
		return graph;
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
	   
	      convBlock(graph, moduleName,moduleIndex, 2, createLayerName(moduleName, ACTIVATION_LAYER,moduleIndex,1), new int[] {3,3}, 32, ConvolutionMode.Truncate);
	      
	      convBlock(graph, moduleName, moduleIndex,3, createLayerName(moduleName, ACTIVATION_LAYER,moduleIndex,2), new int[] {3,3}, 64);
    	 
	      MaxPooling2D(graph,moduleName,moduleIndex,4, createLayerName(moduleName, ACTIVATION_LAYER,moduleIndex,3), new int[] {3, 3}, new int[] {2, 2}, ConvolutionMode.Truncate);
	      
	      convBlock(graph, moduleName, moduleIndex,5, createLayerName(moduleName, ACTIVATION_LAYER,moduleIndex,3), new int[] {3,3},new int[] {2,2},  96, ConvolutionMode.Truncate);
		   
	      graph.addVertex(createLayerName(moduleName, MERGE_VERTEX,moduleIndex,6), new MergeVertex(), new String[]{createLayerName(moduleName, MAX_POOLING,moduleIndex,4),createLayerName(moduleName, ACTIVATION_LAYER,moduleIndex,5)});
	  
    	  //stem right branch start
	      convBlock(graph, moduleName,moduleIndex, 7, createLayerName(moduleName, MERGE_VERTEX,moduleIndex,6), new int[] {1,1}, 64);
	      
	      convBlock(graph, moduleName, moduleIndex,8, createLayerName(moduleName, ACTIVATION_LAYER,moduleIndex,7), new int[] {3,3}, 96, ConvolutionMode.Truncate);
		  
	      //stem right branch end
	      
	      //stem left start
	      convBlock(graph, moduleName, moduleIndex,9, createLayerName(moduleName, MERGE_VERTEX,moduleIndex,6), new int[] {1,1}, 64);
	      convBlock(graph, moduleName,moduleIndex, 10, createLayerName(moduleName, ACTIVATION_LAYER,moduleIndex,9), new int[] {1,7}, 64);
	      convBlock(graph, moduleName, moduleIndex,11, createLayerName(moduleName, ACTIVATION_LAYER,moduleIndex,10), new int[] {7,1}, 64);
	      convBlock(graph, moduleName, moduleIndex,12, createLayerName(moduleName, ACTIVATION_LAYER,moduleIndex,11), new int[] {3,3}, 96, ConvolutionMode.Truncate);
		  //stem left end
	      
	      graph.addVertex(createLayerName(moduleName, MERGE_VERTEX,moduleIndex,13), new MergeVertex(), new String[]{createLayerName(moduleName, ACTIVATION_LAYER,moduleIndex,12),createLayerName(moduleName, ACTIVATION_LAYER,moduleIndex,8)});
    	  
    	  convBlock(graph, moduleName, moduleIndex,14, createLayerName(moduleName, MERGE_VERTEX,moduleIndex,13), new int[] {3,3},new int[] {2,2},  192, ConvolutionMode.Truncate);
		  
	      MaxPooling2D(graph,moduleName,moduleIndex,15, createLayerName(moduleName, MERGE_VERTEX,moduleIndex,13), new int[] {3, 3}, new int[] {2, 2}, ConvolutionMode.Truncate);
		     
          graph.addVertex(createLayerName(moduleName, MERGE_VERTEX,moduleIndex,16), new MergeVertex(), new String[]{createLayerName(moduleName, MAX_POOLING,moduleIndex,15),createLayerName(moduleName, ACTIVATION_LAYER,moduleIndex,14)});
    	  
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

		batchNormAndActivation(graph,input,moduleName,moduleIndex,0);

		//update input name to batchNormAndActivation layer name
		input=createLayerName(moduleName, ACTIVATION_LAYER,moduleIndex,0);

		//branch 1 start
		convBlock(graph, moduleName, moduleIndex,1,input, new int[] {1,1}, 96);

		//branch 2 start
		convBlock(graph, moduleName, moduleIndex,2,input, new int[] {1,1}, 64);
		convBlock(graph, moduleName, moduleIndex,3,createLayerName(moduleName, ACTIVATION_LAYER,moduleIndex,2), new int[] {3,3}, 96);

		//branch 3 start
		convBlock(graph, moduleName, moduleIndex,4,input, new int[] {1,1}, 64);
		convBlock(graph, moduleName, moduleIndex,5,createLayerName(moduleName, ACTIVATION_LAYER,moduleIndex,4), new int[] {3,3}, 96);
		convBlock(graph, moduleName, moduleIndex,6,createLayerName(moduleName, ACTIVATION_LAYER,moduleIndex,5), new int[] {3,3}, 96);


		//branch 4 start

		AveragePooling2D(graph,moduleName,moduleIndex,7,input,new int[] {3,3},new int[] {1,1},ConvolutionMode.Same);
		convBlock(graph, moduleName, moduleIndex,8,createLayerName(moduleName, AVG_POOLING,moduleIndex,7), new int[] {1,1}, 96);


		//merge 4 branches
		graph.addVertex(createLayerName(moduleName, MERGE_VERTEX,moduleIndex,9), new MergeVertex(), new String[]{createLayerName(moduleName, ACTIVATION_LAYER,moduleIndex,1),createLayerName(moduleName, ACTIVATION_LAYER,moduleIndex,3),createLayerName(moduleName, ACTIVATION_LAYER,moduleIndex,6),createLayerName(moduleName, ACTIVATION_LAYER,moduleIndex,8)});

		return graph;

	}


	private ComputationGraphConfiguration.GraphBuilder buildReductionA(ComputationGraphConfiguration.GraphBuilder graph,String input,Integer moduleIndex) {

		String moduleName="reduction-A";
        //start branch 1
		convBlock(graph, moduleName, moduleIndex,0, input, new int[] {3,3},new int[] {2,2},  384, ConvolutionMode.Truncate);

		//start branch 2
		convBlock(graph, moduleName, moduleIndex,1,input, new int[] {1,1}, 192);
		convBlock(graph, moduleName, moduleIndex,2,createLayerName(moduleName, ACTIVATION_LAYER,moduleIndex,1), new int[] {3,3}, 224);
		convBlock(graph, moduleName, moduleIndex,3, createLayerName(moduleName, ACTIVATION_LAYER,moduleIndex,2), new int[] {3,3},new int[] {2,2},  256, ConvolutionMode.Truncate);

		//start branch 3
		MaxPooling2D(graph,moduleName,moduleIndex,4,input,new int[] {3,3},new int[] {2,2},ConvolutionMode.Truncate);


        //merge 3 branches
		graph.addVertex(createLayerName(moduleName, MERGE_VERTEX,moduleIndex,4), new MergeVertex(), new String[]{createLayerName(moduleName, ACTIVATION_LAYER,moduleIndex,0),createLayerName(moduleName, ACTIVATION_LAYER,moduleIndex,3),createLayerName(moduleName, MAX_POOLING,moduleIndex,4)});

		return graph;
	}


	private ComputationGraphConfiguration.GraphBuilder buildInceptionB(ComputationGraphConfiguration.GraphBuilder graph,String input,Integer moduleIndex) {

		String moduleName="inception-B";
        //start branch 1
		convBlock(graph, moduleName, moduleIndex,0,input, new int[] {1,1}, 384);

		//start branch 2
		convBlock(graph, moduleName, moduleIndex,1,input, new int[] {1,1}, 192);
		convBlock(graph, moduleName, moduleIndex,2,createLayerName(moduleName, ACTIVATION_LAYER,moduleIndex,1), new int[] {1,7}, 244);
		convBlock(graph, moduleName, moduleIndex,3,createLayerName(moduleName, ACTIVATION_LAYER,moduleIndex,2), new int[] {7,1}, 256);


		//start branch 3
		convBlock(graph, moduleName, moduleIndex,4,input, new int[] {1,1}, 192);
		convBlock(graph, moduleName, moduleIndex,5,createLayerName(moduleName, ACTIVATION_LAYER,moduleIndex,4), new int[] {7,1}, 192);
		convBlock(graph, moduleName, moduleIndex,6,createLayerName(moduleName, ACTIVATION_LAYER,moduleIndex,5), new int[] {1,7}, 224);
		convBlock(graph, moduleName, moduleIndex,7,createLayerName(moduleName, ACTIVATION_LAYER,moduleIndex,6), new int[] {7,1}, 224);
		convBlock(graph, moduleName, moduleIndex,8,createLayerName(moduleName, ACTIVATION_LAYER,moduleIndex,7), new int[] {1,7}, 256);

		//start branch 4
		AveragePooling2D(graph,moduleName,moduleIndex,9,input,new int[] {3,3},new int[] {1,1},ConvolutionMode.Same);
		convBlock(graph, moduleName, moduleIndex,10,createLayerName(moduleName, AVG_POOLING,moduleIndex,9), new int[] {1,1}, 128);


		//merge 4 branches
		graph.addVertex(createLayerName(moduleName, MERGE_VERTEX,moduleIndex,11), new MergeVertex(), new String[]{createLayerName(moduleName, ACTIVATION_LAYER,moduleIndex,0),createLayerName(moduleName, ACTIVATION_LAYER,moduleIndex,3),createLayerName(moduleName, ACTIVATION_LAYER,moduleIndex,8),createLayerName(moduleName, ACTIVATION_LAYER,moduleIndex,10)});


		return graph;

	}


	private ComputationGraphConfiguration.GraphBuilder buildReductionB(ComputationGraphConfiguration.GraphBuilder graph,String input,Integer moduleIndex) {

		String moduleName="reduction-B";

        //start branch 1
		convBlock(graph, moduleName, moduleIndex,0,input, new int[] {1,1}, 192);
		convBlock(graph, moduleName, moduleIndex,1,  createLayerName(moduleName, ACTIVATION_LAYER,moduleIndex,0), new int[] {3,3},new int[] {2,2},  192, ConvolutionMode.Truncate);

		//start branch 2
		convBlock(graph, moduleName, moduleIndex,2,input, new int[] {1,1}, 256);
		convBlock(graph, moduleName, moduleIndex,3,createLayerName(moduleName, ACTIVATION_LAYER,moduleIndex,2), new int[] {1,7}, 256);
		convBlock(graph, moduleName, moduleIndex,4,createLayerName(moduleName, ACTIVATION_LAYER,moduleIndex,3), new int[] {7,1}, 320);
		convBlock(graph, moduleName, moduleIndex,5,createLayerName(moduleName, ACTIVATION_LAYER,moduleIndex,4), new int[] {3,3},new int[] {2,2},  320, ConvolutionMode.Truncate);


		//start branch 3
		MaxPooling2D(graph,moduleName,moduleIndex,6,input,new int[] {3,3},new int[] {2,2},ConvolutionMode.Truncate);

		//merge 3 branches
		graph.addVertex(createLayerName(moduleName, MERGE_VERTEX,moduleIndex,7), new MergeVertex(), new String[]{createLayerName(moduleName, ACTIVATION_LAYER,moduleIndex,1),createLayerName(moduleName, ACTIVATION_LAYER,moduleIndex,5),createLayerName(moduleName, MAX_POOLING,moduleIndex,6)});

		return graph;
	}


	private ComputationGraphConfiguration.GraphBuilder buildInceptionC(ComputationGraphConfiguration.GraphBuilder graph,String input,Integer moduleIndex) {

		String moduleName="inception-C";
		//start branch 1
		convBlock(graph, moduleName, moduleIndex,0,input, new int[] {1,1}, 256);

		//start branch 2
		convBlock(graph, moduleName, moduleIndex,1,input, new int[] {1,1}, 384);
		convBlock(graph, moduleName, moduleIndex,2,createLayerName(moduleName, ACTIVATION_LAYER,moduleIndex,1), new int[] {1,3}, 256);
		convBlock(graph, moduleName, moduleIndex,3,createLayerName(moduleName, ACTIVATION_LAYER,moduleIndex,1), new int[] {3,1}, 256);

		graph.addVertex(createLayerName(moduleName, MERGE_VERTEX,moduleIndex,4), new MergeVertex(), new String[]{createLayerName(moduleName, ACTIVATION_LAYER,moduleIndex,2),createLayerName(moduleName, ACTIVATION_LAYER,moduleIndex,3)});


		//start branch 3
		convBlock(graph, moduleName, moduleIndex,5,input, new int[] {1,1}, 384);
		convBlock(graph, moduleName, moduleIndex,6,createLayerName(moduleName, ACTIVATION_LAYER,moduleIndex,5), new int[] {3,1}, 448);
		convBlock(graph, moduleName, moduleIndex,7,createLayerName(moduleName, ACTIVATION_LAYER,moduleIndex,6), new int[] {1,3}, 512);

		convBlock(graph, moduleName, moduleIndex,8,createLayerName(moduleName, ACTIVATION_LAYER,moduleIndex,7), new int[] {1,3}, 256);
		convBlock(graph, moduleName, moduleIndex,9,createLayerName(moduleName, ACTIVATION_LAYER,moduleIndex,7), new int[] {3,1}, 256);

		graph.addVertex(createLayerName(moduleName, MERGE_VERTEX,moduleIndex,10), new MergeVertex(), new String[]{createLayerName(moduleName, ACTIVATION_LAYER,moduleIndex,8),createLayerName(moduleName, ACTIVATION_LAYER,moduleIndex,9)});


		//start branch 4
		AveragePooling2D(graph,moduleName,moduleIndex,11,input,new int[] {3,3},new int[] {1,1},ConvolutionMode.Same);
		convBlock(graph, moduleName, moduleIndex,12,createLayerName(moduleName, AVG_POOLING,moduleIndex,11), new int[] {1,1}, 256);


		//merge 4 branches
		graph.addVertex(createLayerName(moduleName, MERGE_VERTEX,moduleIndex,13), new MergeVertex(), new String[]{createLayerName(moduleName, ACTIVATION_LAYER,moduleIndex,0),createLayerName(moduleName, MERGE_VERTEX,moduleIndex,4),createLayerName(moduleName, MERGE_VERTEX,moduleIndex,10),createLayerName(moduleName, ACTIVATION_LAYER,moduleIndex,12)});


		return graph;

	}

	/**
	 * one model has one or more
	 * module,one module has one
	 * or more block,so the name of
	 * layer is constructed with
	 * moduleName+"-"+layerName+"-"+moduleIndex+"-"+blockIndex
	 * @param moduleName
	 * @param layerName
	 * @param moduleIndex
	 * @param blockIndex
	 * @return
	 */
	private String createLayerName(String moduleName, String layerName,Integer moduleIndex,Integer blockIndex) {
		String newLayerName=moduleName.concat("-").concat(layerName).concat("-").concat(String.valueOf(moduleIndex)).concat("-").concat(String.valueOf(blockIndex));
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

		String batchNormAndActivationInput=createLayerName(moduleName,"cnn",moduleIndex,blockIndex);

		batchNormAndActivation(graph,batchNormAndActivationInput, moduleName,moduleIndex, blockIndex);

		return graph;
	}

	private void batchNormAndActivation(ComputationGraphConfiguration.GraphBuilder graph,String batchNormAndActivationInput, String moduleName,int moduleIndex,int blockIndex) {
		graph.addLayer(createLayerName(moduleName,"batch",moduleIndex,blockIndex),
				new BatchNormalization.Builder(false)
						.decay(0.99)
						.eps(0.001)
						.build(),
				batchNormAndActivationInput);
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

	@Override
    public ModelMetaData metaData() {
        return new ModelMetaData(new int[][] {inputShape}, 1, ZooType.CNN);
    }

    @Override
    public void setInputShape(int[][] inputShape) {
        this.inputShape = inputShape[0];
    }

}
