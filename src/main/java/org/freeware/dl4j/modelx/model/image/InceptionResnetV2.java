package org.freeware.dl4j.modelx.model.image;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.*;
import org.deeplearning4j.nn.conf.distribution.TruncatedNormalDistribution;
import org.deeplearning4j.nn.conf.graph.MergeVertex;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.conf.preprocessor.CnnToFeedForwardPreProcessor;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.zoo.ModelMetaData;
import org.deeplearning4j.zoo.PretrainedType;
import org.deeplearning4j.zoo.ZooModel;
import org.deeplearning4j.zoo.ZooType;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.IUpdater;
import org.nd4j.linalg.learning.config.RmsProp;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.HashMap;
import java.util.Map;


/**
 * 
 * https://arxiv.org/pdf/1602.07261v1.pdf
 * @author wenfengxu
 *
 */
@AllArgsConstructor
@Builder
@Slf4j
public class InceptionResnetV2 extends ZooModel {

    @Builder.Default private long seed = 1234;
    @Builder.Default private int[] inputShape = new int[] {3, 299, 299};
    @Builder.Default private int numClasses = 0;
    @Builder.Default private IUpdater updater = new RmsProp(0.1, 0.96, 0.001);
    @Builder.Default private CacheMode cacheMode = CacheMode.NONE;
    @Builder.Default private WorkspaceMode workspaceMode = WorkspaceMode.ENABLED;
    @Builder.Default private ConvolutionLayer.AlgoMode cudnnAlgoMode = ConvolutionLayer.AlgoMode.PREFER_FASTEST;



	private static String ACTIVATION_LAYER ="activation-layer";

	private static String MERGE_VERTEX ="merge-vertex";

	private static String MAX_POOLING ="max-pooling";

	private static String AVG_POOLING ="avg-pooling";

	private static String CNN ="cnn";

    private InceptionResnetV2() {

	}

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
      

        
        ComputationGraphConfiguration.GraphBuilder graphBuilder = graphBuilder("input");

        String input=createLayerName("stem", ACTIVATION_LAYER,0,16);

        graphBuilder.addInputs("input").setInputTypes(InputType.convolutional(inputShape[2], inputShape[1], inputShape[0]))
        


                        .addLayer("outputLayer",new OutputLayer.Builder().nOut(numClasses)
										.lossFunction(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
										.activation(Activation.SOFTMAX).build()

                                                      ,
								input)

                        .setOutputs("outputLayer");

        ComputationGraphConfiguration conf = graphBuilder.build();



        ComputationGraph model = new ComputationGraph(conf);

        model.init();

        return model;
    }


	public ComputationGraphConfiguration.GraphBuilder graphBuilder(String input) {

		ComputationGraphConfiguration.GraphBuilder graphBuilder
				= new NeuralNetConfiguration.Builder().seed(seed)
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


		graphBuilder=buildInceptionStem(graphBuilder, input);




		return graphBuilder;
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
		//c
		convBlock(graph, moduleName, moduleIndex,0, input, new int[] {3,3},new int[] {2,2}, inputShape[0], 32, ConvolutionMode.Truncate);
		//c
		convBlock(graph, moduleName, moduleIndex,1, createLayerName(moduleName, CNN,moduleIndex,0), new int[] {3,3}, 32, ConvolutionMode.Truncate);
        //c
		convBlock(graph, moduleName, moduleIndex,2, createLayerName(moduleName, CNN,moduleIndex,1), new int[] {3,3}, 64, ConvolutionMode.Same);

        //c1
		MaxPooling2D(graph,moduleName,moduleIndex,3, createLayerName(moduleName, CNN,moduleIndex,2), new int[] {3, 3}, new int[] {2, 2}, ConvolutionMode.Truncate);
        //c2
		convBlock(graph, moduleName, moduleIndex,4, createLayerName(moduleName, CNN,moduleIndex,2), new int[] {3,3},new int[] {2,2}, 96, ConvolutionMode.Truncate);

		//m
		graph.addVertex(createLayerName(moduleName, MERGE_VERTEX,moduleIndex,5), new MergeVertex(), new String[]{createLayerName(moduleName, MAX_POOLING,moduleIndex,3),createLayerName(moduleName, CNN,moduleIndex,4)});

		//c1
		convBlock(graph, moduleName, moduleIndex,6, createLayerName(moduleName, MERGE_VERTEX,moduleIndex,5), new int[] {1,1}, 64, ConvolutionMode.Same);
		//c1
		convBlock(graph, moduleName, moduleIndex,7, createLayerName(moduleName, CNN,moduleIndex,6), new int[] {3,3}, 96, ConvolutionMode.Truncate);

		//c2
		convBlock(graph, moduleName, moduleIndex,8, createLayerName(moduleName, MERGE_VERTEX,moduleIndex,5), new int[] {1,1}, 64, ConvolutionMode.Same);
		//c2
		convBlock(graph, moduleName, moduleIndex,9, createLayerName(moduleName, CNN,moduleIndex,8), new int[] {7,1}, 64, ConvolutionMode.Same);
		//c2
		convBlock(graph, moduleName, moduleIndex,10, createLayerName(moduleName, CNN,moduleIndex,9), new int[] {1,7}, 64, ConvolutionMode.Same);
		//c2
		convBlock(graph, moduleName, moduleIndex,11, createLayerName(moduleName, CNN,moduleIndex,10), new int[] {3,3}, 96, ConvolutionMode.Truncate);

		//m2
		graph.addVertex(createLayerName(moduleName, MERGE_VERTEX,moduleIndex,12), new MergeVertex(), new String[]{createLayerName(moduleName, CNN,moduleIndex,7),createLayerName(moduleName, CNN,moduleIndex,11)});

        //p1
		MaxPooling2D(graph,moduleName,moduleIndex,13, createLayerName(moduleName, MERGE_VERTEX,moduleIndex,12), new int[] {3, 3}, new int[] {2, 2}, ConvolutionMode.Truncate);
		//p2
		convBlock(graph, moduleName, moduleIndex,14, createLayerName(moduleName, MERGE_VERTEX,moduleIndex,12), new int[] {3,3},new int[] {2,2}, 192, ConvolutionMode.Truncate);

		//m3
		graph.addVertex(createLayerName(moduleName, MERGE_VERTEX,moduleIndex,15), new MergeVertex(), new String[]{createLayerName(moduleName, MAX_POOLING,moduleIndex,13),createLayerName(moduleName, CNN,moduleIndex,14)});

		batchNormAndActivation(graph,createLayerName(moduleName, MERGE_VERTEX,moduleIndex,15), moduleName,moduleIndex,16);

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


		return graph;

	}


	private ComputationGraphConfiguration.GraphBuilder buildReductionA(ComputationGraphConfiguration.GraphBuilder graph,String input,Integer moduleIndex) {

		String moduleName="reduction-A";

		return graph;
	}


	private ComputationGraphConfiguration.GraphBuilder buildInceptionB(ComputationGraphConfiguration.GraphBuilder graph,String input,Integer moduleIndex) {

		String moduleName="inception-B";


		return graph;

	}


	private ComputationGraphConfiguration.GraphBuilder buildReductionB(ComputationGraphConfiguration.GraphBuilder graph,String input,Integer moduleIndex) {

		String moduleName="reduction-B";


		return graph;
	}


	private ComputationGraphConfiguration.GraphBuilder buildInceptionC(ComputationGraphConfiguration.GraphBuilder graph,String input,Integer moduleIndex) {

		String moduleName="inception-C";

		return graph;

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

		graph.addLayer(createLayerName(moduleName,CNN,moduleIndex,blockIndex),
				builder
						.nOut(out)
						.convolutionMode(convolutionMode)
						.cudnnAlgoMode(cudnnAlgoMode)
						.build(),
				input);



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
