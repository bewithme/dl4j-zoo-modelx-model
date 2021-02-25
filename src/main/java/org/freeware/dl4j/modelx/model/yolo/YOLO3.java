package org.freeware.dl4j.modelx.model.yolo;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.*;
import org.deeplearning4j.nn.conf.graph.ElementWiseVertex;
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
import org.nd4j.linalg.lossfunctions.LossFunctions;


/**
 *
 *
 *
 * @author wenfengxu  wechatid:italybaby
 *
 */
@AllArgsConstructor
@Builder
@Slf4j
public class YOLO3 extends ZooModel {

    @Builder.Default private long seed = 1234;
    @Builder.Default private int[] inputShape = new int[] {3, 416, 416};
    @Builder.Default private int numClasses = 0;
    @Builder.Default private IUpdater updater = new Adam(0.001);
    @Builder.Default private CacheMode cacheMode = CacheMode.NONE;
    @Builder.Default private WorkspaceMode workspaceMode = WorkspaceMode.ENABLED;
    @Builder.Default private ConvolutionLayer.AlgoMode cudnnAlgoMode = ConvolutionLayer.AlgoMode.PREFER_FASTEST;

    
   private static String ACTIVATION ="activation";
    
   private static String ELEMENT_WISE_VERTEX ="element-wise-vertex";

	private static String MERGE_VERTEX ="merge-vertex";
    
   private static String MAX_POOLING ="max-pooling";

   private static String AVG_POOLING ="avg-pooling";

   private static String UP_SAMPLING_2D="up-sampling-2d";

   private static String ZERO_PADDING ="zero-padding";

   private static String CNN ="cnn";

   private static String BATCH_NORM="batch-norm";

   private YOLO3() {}

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

   	    ComputationGraphConfiguration.GraphBuilder graph = graphBuilder();

        String lastLayerName= buildLayerZeroToFour(graph,"input");

        lastLayerName=buildLayerFiveToEight(graph,lastLayerName);

        lastLayerName=buildLayerNineToEleven(graph,lastLayerName);

        lastLayerName=buildLayerTwelveToFifteen(graph,lastLayerName);

        //skip 36
        lastLayerName=buildLayerSixteenToThirtySix(graph,lastLayerName);

        String skipThirtySix=lastLayerName;

        lastLayerName=buildLayerThirtySevenToForty(graph,lastLayerName);

        //skip 61
        lastLayerName=buildLayerFortyOneToSixtyOne(graph,lastLayerName);

        String skipSixtyOne=lastLayerName;

        lastLayerName=buildLayerSixtyTwoToSixtyFive(graph,lastLayerName);

        lastLayerName=buildLayerSixtySixToSeventyFour(graph,lastLayerName);

        lastLayerName=buildLayerSeventyFiveToSeventyNine(graph,lastLayerName);

        String beforePredictLayerName=lastLayerName;

        String predictSmallLayerName=buildLayerPredictSmall(graph,beforePredictLayerName);

        String eightyThreeToEightySixLayerName=buildLayerEightyThreeToEightySix(graph,beforePredictLayerName,skipSixtyOne);

        graph.addInputs("input").setInputTypes(InputType.convolutional(inputShape[2], inputShape[1], inputShape[0]))

				.addLayer("outputLayer",new OutputLayer.Builder().nOut(numClasses)
								.lossFunction(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
								.activation(Activation.SOFTMAX)
								.build()
						,
						new String[]{predictSmallLayerName,eightyThreeToEightySixLayerName})
                        .setOutputs("outputLayer");

        ComputationGraphConfiguration conf = graph.build();

        ComputationGraph model = new ComputationGraph(conf);

        model.init();

        return model;
    }


	public ComputationGraphConfiguration.GraphBuilder graphBuilder() {

		ComputationGraphConfiguration.GraphBuilder graph = new NeuralNetConfiguration.Builder().seed(seed)
				.activation(Activation.IDENTITY)
				.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
				.updater(updater)
				.weightInit(WeightInit.XAVIER)
				.miniBatch(true)
				.cacheMode(cacheMode)
				.trainingWorkspaceMode(workspaceMode)
				.inferenceWorkspaceMode(workspaceMode)
				.convolutionMode(ConvolutionMode.Truncate)
				.graphBuilder();

		return graph;
	}




	/**
	 * Layer  0 => 4
	 * @param graph
	 * @param input
	 * @return
	 */
	private String buildLayerZeroToFour(ComputationGraphConfiguration.GraphBuilder graph,String input) {

		String moduleName="layer-zero-to-four";

		int moduleIndex=0;

		String firstLayerName=convBlock(graph, moduleName, moduleIndex,0, input, new int[] {3,3},new int[] {1,1}, inputShape[0],32, ConvolutionMode.Same, Boolean.TRUE,Boolean.TRUE);

		String secondLayerName=convBlock(graph, moduleName, moduleIndex,1, firstLayerName, new int[] {3,3},new int[] {2,2},64, ConvolutionMode.Truncate, Boolean.TRUE,Boolean.TRUE);

		String thirdLayerName=convBlock(graph, moduleName, moduleIndex,2, secondLayerName, new int[] {1,1},new int[] {1,1},32, ConvolutionMode.Same, Boolean.TRUE,Boolean.TRUE);

		String fourthLayerName=convBlock(graph, moduleName, moduleIndex,3, thirdLayerName, new int[] {3,3},new int[] {1,1},64, ConvolutionMode.Same, Boolean.TRUE,Boolean.TRUE);

		String fifthLayer=createLayerName(moduleName, ELEMENT_WISE_VERTEX,moduleIndex,4);

		graph.addVertex(fifthLayer, new ElementWiseVertex(ElementWiseVertex.Op.Add), new String[]{fourthLayerName,secondLayerName});

		return fifthLayer;
	}

	/**
	 * Layer 5 => 8
	 * @param graph
	 * @param input
	 * @return
	 */
	private String buildLayerFiveToEight(ComputationGraphConfiguration.GraphBuilder graph,String input) {

		String moduleName="layer-five-to-eight";

		int moduleIndex=0;

		String firstLayerName=convBlock(graph, moduleName, moduleIndex,0, input, new int[] {3,3},new int[] {2,2},128, ConvolutionMode.Truncate, Boolean.TRUE,Boolean.TRUE);

		String secondLayerName=convBlock(graph, moduleName, moduleIndex,1, firstLayerName, new int[] {1,1},new int[] {1,1},64, ConvolutionMode.Same, Boolean.TRUE,Boolean.TRUE);

		String thirdLayerName=convBlock(graph, moduleName, moduleIndex,2, secondLayerName, new int[] {3,3},new int[] {1,1},128, ConvolutionMode.Same, Boolean.TRUE,Boolean.TRUE);

		String fourthLayerName=createLayerName(moduleName, ELEMENT_WISE_VERTEX,moduleIndex,3);

		graph.addVertex(fourthLayerName, new ElementWiseVertex(ElementWiseVertex.Op.Add), new String[]{thirdLayerName,firstLayerName});

		return fourthLayerName;
	}

	/**
	 * Layer  9 => 11
	 * @param graph
	 * @param input
	 * @return
	 */
	private String buildLayerNineToEleven(ComputationGraphConfiguration.GraphBuilder graph,String input) {

		String moduleName="layer-nine-to-eleven";

		int moduleIndex=0;

		String firstLayerName=convBlock(graph, moduleName, moduleIndex,0, input, new int[] {1,1},new int[] {1,1},64, ConvolutionMode.Same, Boolean.TRUE,Boolean.TRUE);

		String secondLayerName=convBlock(graph, moduleName, moduleIndex,1, firstLayerName, new int[] {3,3},new int[] {1,1},128, ConvolutionMode.Same, Boolean.TRUE,Boolean.TRUE);

		String thirdLayerName=createLayerName(moduleName, ELEMENT_WISE_VERTEX,moduleIndex,2);

		graph.addVertex(thirdLayerName, new ElementWiseVertex(ElementWiseVertex.Op.Add), new String[]{secondLayerName,input});

		return thirdLayerName;
	}

	/**
	 * Layer 12 => 15
	 * @param graph
	 * @param input
	 * @return
	 */
	private String buildLayerTwelveToFifteen(ComputationGraphConfiguration.GraphBuilder graph,String input) {

		String moduleName="layer-twelve-to-fifteen";

		int moduleIndex=0;

		String firstLayerName=convBlock(graph, moduleName, moduleIndex,0, input, new int[] {3,3},new int[] {2,2},256, ConvolutionMode.Truncate, Boolean.TRUE,Boolean.TRUE);

		String secondLayerName=convBlock(graph, moduleName, moduleIndex,1, firstLayerName, new int[] {1,1},new int[] {1,1},128, ConvolutionMode.Same, Boolean.TRUE,Boolean.TRUE);

		String thirdLayerName=convBlock(graph, moduleName, moduleIndex,2, secondLayerName, new int[] {3,3},new int[] {1,1},256, ConvolutionMode.Same, Boolean.TRUE,Boolean.TRUE);

		String fourthLayerName=createLayerName(moduleName, ELEMENT_WISE_VERTEX,moduleIndex,3);

		graph.addVertex(fourthLayerName, new ElementWiseVertex(ElementWiseVertex.Op.Add), new String[]{thirdLayerName,firstLayerName});

		return fourthLayerName;
	}

	/**
	 * item of layer 16 => 36
	 * @param graph
	 * @param input
	 * @return
	 */
	private String buildLayerSixteenToThirtySixItem(ComputationGraphConfiguration.GraphBuilder graph,String input,int moduleIndex) {

		String moduleName="layer-sixteen-to-thirty-six";

		String firstLayerName=convBlock(graph, moduleName, moduleIndex,0, input, new int[] {1,1},new int[] {1,1},128, ConvolutionMode.Same, Boolean.TRUE,Boolean.TRUE);

		String secondLayerName=convBlock(graph, moduleName, moduleIndex,1, firstLayerName, new int[] {3,3},new int[] {1,1},256, ConvolutionMode.Same, Boolean.TRUE,Boolean.TRUE);

		String thirdLayerName=createLayerName(moduleName, ELEMENT_WISE_VERTEX,moduleIndex,2);

		graph.addVertex(thirdLayerName, new ElementWiseVertex(ElementWiseVertex.Op.Add), new String[]{secondLayerName,input});

		return thirdLayerName;
	}

	/**
	 * layer 16 => 36
	 * @param graph
	 * @param input
	 * @return
	 */
	private String buildLayerSixteenToThirtySix(ComputationGraphConfiguration.GraphBuilder graph,String input) {

		String lastLayerName=null;

		for(int moduleIndex=0;moduleIndex<7;moduleIndex++){

			lastLayerName=buildLayerSixteenToThirtySixItem(graph,input,moduleIndex);

			input=lastLayerName;
		}

		return lastLayerName;
	}


	/**
	 * Layer 37 => 40
	 * @param graph
	 * @param input
	 * @return
	 */
	private String buildLayerThirtySevenToForty(ComputationGraphConfiguration.GraphBuilder graph,String input) {

		String moduleName="layer-thirty-seven-to-forty";

		int moduleIndex=0;

		String firstLayerName=convBlock(graph, moduleName, moduleIndex,0, input, new int[] {3,3},new int[] {2,2},512, ConvolutionMode.Truncate, Boolean.TRUE,Boolean.TRUE);

		String secondLayerName=convBlock(graph, moduleName, moduleIndex,1, firstLayerName, new int[] {1,1},new int[] {1,1},256, ConvolutionMode.Same, Boolean.TRUE,Boolean.TRUE);

		String thirdLayerName=convBlock(graph, moduleName, moduleIndex,2, secondLayerName, new int[] {3,3},new int[] {1,1},512, ConvolutionMode.Same, Boolean.TRUE,Boolean.TRUE);

		String fourthLayerName=createLayerName(moduleName, ELEMENT_WISE_VERTEX,moduleIndex,3);

		graph.addVertex(fourthLayerName, new ElementWiseVertex(ElementWiseVertex.Op.Add), new String[]{thirdLayerName,firstLayerName});

		return fourthLayerName;
	}



	/**
	 * item of Layer 41 => 61
	 * @param graph
	 * @param input
	 * @return
	 */
	private String buildLayerFortyOneToSixtyOneItem(ComputationGraphConfiguration.GraphBuilder graph,String input,int moduleIndex) {

		String moduleName="layer-forty-one-to-sixty-one";

		String firstLayerName=convBlock(graph, moduleName, moduleIndex,0, input, new int[] {1,1},new int[] {1,1},256, ConvolutionMode.Same, Boolean.TRUE,Boolean.TRUE);

		String secondLayerName=convBlock(graph, moduleName, moduleIndex,1, firstLayerName, new int[] {3,3},new int[] {1,1},512, ConvolutionMode.Same, Boolean.TRUE,Boolean.TRUE);

		String thirdLayerName=createLayerName(moduleName, ELEMENT_WISE_VERTEX,moduleIndex,2);

		graph.addVertex(thirdLayerName, new ElementWiseVertex(ElementWiseVertex.Op.Add), new String[]{secondLayerName,input});

		return thirdLayerName;
	}

	/**
	 * Layer 41 => 61
	 * @param graph
	 * @param input
	 * @return
	 */
	private String buildLayerFortyOneToSixtyOne(ComputationGraphConfiguration.GraphBuilder graph,String input) {

		String lastLayerName=null;

		for(int moduleIndex=0;moduleIndex<7;moduleIndex++){

			lastLayerName=buildLayerFortyOneToSixtyOneItem(graph,input,moduleIndex);

			input=lastLayerName;
		}

		return lastLayerName;
	}




	/**
	 * Layer 62 => 65
	 * @param graph
	 * @param input
	 * @return
	 */
	private String buildLayerSixtyTwoToSixtyFive(ComputationGraphConfiguration.GraphBuilder graph,String input) {

		String moduleName="layer-sixty-two-to-sixty-five";

		int moduleIndex=0;

		String firstLayerName=convBlock(graph, moduleName, moduleIndex,0, input, new int[] {3,3},new int[] {2,2},1024, ConvolutionMode.Truncate, Boolean.TRUE,Boolean.TRUE);

		String secondLayerName=convBlock(graph, moduleName, moduleIndex,1, firstLayerName, new int[] {1,1},new int[] {1,1},512, ConvolutionMode.Same, Boolean.TRUE,Boolean.TRUE);

		String thirdLayerName=convBlock(graph, moduleName, moduleIndex,2, secondLayerName, new int[] {3,3},new int[] {1,1},1024, ConvolutionMode.Same, Boolean.TRUE,Boolean.TRUE);

		String fourthLayerName=createLayerName(moduleName, ELEMENT_WISE_VERTEX,moduleIndex,3);

		graph.addVertex(fourthLayerName, new ElementWiseVertex(ElementWiseVertex.Op.Add), new String[]{thirdLayerName,firstLayerName});

		return fourthLayerName;
	}


	/**
	 * item of Layer 66 => 74
	 * @param graph
	 * @param input
	 * @return
	 */
	private String buildLayerSixtySixToSeventyFourItem(ComputationGraphConfiguration.GraphBuilder graph,String input,int moduleIndex) {

		String moduleName="layer-sixty-six-to-seventy-four";

		String firstLayerName=convBlock(graph, moduleName, moduleIndex,0, input, new int[] {1,1},new int[] {1,1},512, ConvolutionMode.Same, Boolean.TRUE,Boolean.TRUE);

		String secondLayerName=convBlock(graph, moduleName, moduleIndex,1, firstLayerName, new int[] {3,3},new int[] {1,1},1024, ConvolutionMode.Same, Boolean.TRUE,Boolean.TRUE);

		String thirdLayerName=createLayerName(moduleName, ELEMENT_WISE_VERTEX,moduleIndex,2);

		graph.addVertex(thirdLayerName, new ElementWiseVertex(ElementWiseVertex.Op.Add), new String[]{secondLayerName,input});

		return thirdLayerName;
	}

	/**
	 * Layer 66 => 74
	 * @param graph
	 * @param input
	 * @return
	 */
	private String buildLayerSixtySixToSeventyFour(ComputationGraphConfiguration.GraphBuilder graph,String input) {

		String lastLayerName=null;

		for(int moduleIndex=0;moduleIndex<3;moduleIndex++){

			lastLayerName=buildLayerSixtySixToSeventyFourItem(graph,input,moduleIndex);

			input=lastLayerName;
		}

		return lastLayerName;
	}


	/**
	 * Layer 75 => 79
	 * @param graph
	 * @param input
	 * @return
	 */
	private String buildLayerSeventyFiveToSeventyNine(ComputationGraphConfiguration.GraphBuilder graph,String input) {


		String moduleName="layer-seventy-five-to-seventy-nine";

		int moduleIndex=0;

		input=convBlock(graph, moduleName, moduleIndex,0, input, new int[] {1,1},new int[] {1,1},512, ConvolutionMode.Same, Boolean.TRUE,Boolean.TRUE);

		input=convBlock(graph, moduleName, moduleIndex,1, input, new int[] {3,3},new int[] {1,1},1024, ConvolutionMode.Same, Boolean.TRUE,Boolean.TRUE);

		input=convBlock(graph, moduleName, moduleIndex,2, input, new int[] {1,1},new int[] {1,1},512, ConvolutionMode.Same, Boolean.TRUE,Boolean.TRUE);

		input=convBlock(graph, moduleName, moduleIndex,3, input, new int[] {3,3},new int[] {1,1},1024, ConvolutionMode.Same, Boolean.TRUE,Boolean.TRUE);

		input=convBlock(graph, moduleName, moduleIndex,4, input, new int[] {1,1},new int[] {1,1},512, ConvolutionMode.Same, Boolean.TRUE,Boolean.TRUE);


		return input;
	}

	/**
	 * Layer 80 => 82
	 * @param graph
	 * @param input
	 * @return
	 */
	private String buildLayerPredictSmall(ComputationGraphConfiguration.GraphBuilder graph,String input) {

		String moduleName="layer-eighty-to-eighty-two";

		int moduleIndex=0;

		String firstLayerName=convBlock(graph, moduleName, moduleIndex,0, input, new int[] {3,3},new int[] {1,1},1024, ConvolutionMode.Same, Boolean.TRUE,Boolean.TRUE);

		String secondLayerName=convBlock(graph, moduleName, moduleIndex,1, firstLayerName, new int[] {1,1},new int[] {1,1},(3*(5+numClasses)), ConvolutionMode.Same, Boolean.TRUE,Boolean.FALSE);

		return secondLayerName;
	}


	private String buildLayerEightyThreeToEightySix(ComputationGraphConfiguration.GraphBuilder graph,String input,String skipLayerName) {

		String moduleName="layer-eighty-three-to-eighty-six";

		int moduleIndex=0;

		input=convBlock(graph, moduleName, moduleIndex,0, input, new int[] {1,1},new int[] {1,1},256, ConvolutionMode.Same, Boolean.TRUE,Boolean.TRUE);

        input=upSampling2D(graph,moduleName,moduleIndex,1,input);

		String thirdLayer=createLayerName(moduleName, MERGE_VERTEX,moduleIndex,2);

		graph.addVertex(thirdLayer, new MergeVertex(), new String[]{input,skipLayerName});

		return thirdLayer;
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

	private String convBlock(ComputationGraphConfiguration.GraphBuilder graph,String moduleName,int moduleIndex,int blockIndex,String input,int[] kernelSize, int[] stride,int in,int out,ConvolutionMode convolutionMode,Boolean addBatchNorm,Boolean addLeakyRelu) {

		ConvolutionLayer.Builder builder=new ConvolutionLayer.Builder(
				kernelSize,
				stride);
		if(in>0){
			builder.nIn(in);
		}

		if(stride[0]>1){

            String zeroPaddingLayerName=createLayerName(moduleName, ZERO_PADDING,moduleIndex,blockIndex);

            graph.addLayer(zeroPaddingLayerName,
					new ZeroPaddingLayer.Builder(1,0,1,0).build()
							,
					input);

			input=zeroPaddingLayerName;
		}

        String cnnLayerName=createLayerName(moduleName, CNN,moduleIndex,blockIndex);

		graph.addLayer(cnnLayerName,
				builder
						.nOut(out)
						.convolutionMode(convolutionMode)
						.cudnnAlgoMode(cudnnAlgoMode)
						.build(),
				input);

		input=cnnLayerName;

		String lastLayerName=cnnLayerName;

		if(addBatchNorm==Boolean.TRUE) {

			String batchNormLayerName = createLayerName(moduleName,BATCH_NORM,moduleIndex,blockIndex);

			graph.addLayer(batchNormLayerName,
					new BatchNormalization.Builder(false)
							.decay(0.99)
							.eps(0.001)
							.build(),
					input);

			input=batchNormLayerName;

			lastLayerName=batchNormLayerName;

		}

		if(addLeakyRelu==Boolean.TRUE){
			String leakyLayerName=createLayerName(moduleName,ACTIVATION,moduleIndex,blockIndex);
			graph.addLayer(leakyLayerName,
					new ActivationLayer.Builder()
							.activation(Activation.LEAKYRELU)
							.build(), input);
			lastLayerName=leakyLayerName;
		}

		return lastLayerName;
	}


	private String convBlock(ComputationGraphConfiguration.GraphBuilder graph,String moduleName,int moduleIndex,int blockIndex,String input,int[] kernelSize, int[] stride,int out,ConvolutionMode convolutionMode,Boolean addBatchNorm,Boolean addLeakyRelu) {
		int in=0;
		return convBlock(graph, moduleName, moduleIndex,blockIndex, input, kernelSize, stride,in, out, convolutionMode,addBatchNorm,addLeakyRelu);
	}



	private String upSampling2D(ComputationGraphConfiguration.GraphBuilder graph, String moduleName, Integer moduleIndex, Integer blockIndex,String input) {

		String layerName=createLayerName(moduleName,UP_SAMPLING_2D,moduleIndex,blockIndex);

		graph.addLayer(layerName,
						new Upsampling2D.Builder(2).build(),
						input);
		return layerName;
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
