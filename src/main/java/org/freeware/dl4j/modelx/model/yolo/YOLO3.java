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

        String stemLastLayerName= buildLayerFromZeroToFour(graph,"input");

        stemLastLayerName= buildLayerFromFiveToEight(graph,stemLastLayerName);

        stemLastLayerName= buildLayerFromNineToEleven(graph,stemLastLayerName);

        stemLastLayerName= buildLayerFromTwelveToFifteen(graph,stemLastLayerName);

        //skip 36
        stemLastLayerName= buildLayerFromSixteenToThirtySix(graph,stemLastLayerName);

        String skipThirtySix=stemLastLayerName;

        stemLastLayerName= buildLayerFromThirtySevenToForty(graph,stemLastLayerName);

        //skip 61
        stemLastLayerName= buildLayerFromFortyOneToSixtyOne(graph,stemLastLayerName);

        String skipSixtyOne=stemLastLayerName;

        stemLastLayerName= buildLayerFromSixtyTwoToSixtyFive(graph,stemLastLayerName);

        stemLastLayerName= buildLayerFromSixtySixToSeventyFour(graph,stemLastLayerName);

        stemLastLayerName= buildLayerFromSeventyFiveToSeventyNine(graph,stemLastLayerName);

        //small objects
        String smallObjectDetectionLayerName= buildLayerSmallObjectDetection(graph,stemLastLayerName);

        String predictMediumLastLayerName= buildLayerFromEightyThreeToEightySix(graph,stemLastLayerName,skipSixtyOne);

		predictMediumLastLayerName= buildLayerFromEightySevenToNinetyOne(graph,predictMediumLastLayerName);
        //medium objects
		String mediumObjectDetectionLayerName= buildLayerForMediumObjectDetection(graph,predictMediumLastLayerName);


		String predictBigLastLayerName= buildLayerFromNinetyFiveToNinetyEight(graph,predictMediumLastLayerName,skipThirtySix);

		//big objects
		String predictBigLayerName= buildLayerForBigObjectDetection(graph,predictBigLastLayerName);


		graph.addInputs("input").setInputTypes(InputType.convolutional(inputShape[2], inputShape[1], inputShape[0]))

				.addLayer("outputLayer",new OutputLayer.Builder().nOut(numClasses)
								.lossFunction(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
								.activation(Activation.SOFTMAX)
								.build()
						,
						smallObjectDetectionLayerName)
				.addLayer("outputLayer1",new OutputLayer.Builder().nOut(numClasses)
								.lossFunction(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
								.activation(Activation.SOFTMAX)
								.build()
						,
						mediumObjectDetectionLayerName)
				.addLayer("outputLayer2",new OutputLayer.Builder().nOut(numClasses)
								.lossFunction(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
								.activation(Activation.SOFTMAX)
								.build()
						,
						predictBigLayerName)
                        .setOutputs("outputLayer","outputLayer1","outputLayer2");

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
	private String buildLayerFromZeroToFour(ComputationGraphConfiguration.GraphBuilder graph, String input) {

		String moduleName="layer-0-to-4";

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
	private String buildLayerFromFiveToEight(ComputationGraphConfiguration.GraphBuilder graph, String input) {

		String moduleName="layer-5-to-8";

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
	private String buildLayerFromNineToEleven(ComputationGraphConfiguration.GraphBuilder graph, String input) {

		String moduleName="layer-9-to-11";

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
	private String buildLayerFromTwelveToFifteen(ComputationGraphConfiguration.GraphBuilder graph, String input) {

		String moduleName="layer-12-to-15";

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

		String moduleName="layer-16-to-36";

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
	private String buildLayerFromSixteenToThirtySix(ComputationGraphConfiguration.GraphBuilder graph, String input) {

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
	private String buildLayerFromThirtySevenToForty(ComputationGraphConfiguration.GraphBuilder graph, String input) {

		String moduleName="layer-37-to-40";

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

		String moduleName="layer-41-to-61";

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
	private String buildLayerFromFortyOneToSixtyOne(ComputationGraphConfiguration.GraphBuilder graph, String input) {

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
	private String buildLayerFromSixtyTwoToSixtyFive(ComputationGraphConfiguration.GraphBuilder graph, String input) {

		String moduleName="layer-62-to-65";

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

		String moduleName="layer-66-to-74";

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
	private String buildLayerFromSixtySixToSeventyFour(ComputationGraphConfiguration.GraphBuilder graph, String input) {

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
	private String buildLayerFromSeventyFiveToSeventyNine(ComputationGraphConfiguration.GraphBuilder graph, String input) {


		String moduleName="layer-75-to-79";

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
	private String buildLayerSmallObjectDetection(ComputationGraphConfiguration.GraphBuilder graph, String input) {

		String moduleName="layer-80-to-82";

		int moduleIndex=0;

		String firstLayerName=convBlock(graph, moduleName, moduleIndex,0, input, new int[] {3,3},new int[] {1,1},1024, ConvolutionMode.Same, Boolean.TRUE,Boolean.TRUE);

		String secondLayerName=convBlock(graph, moduleName, moduleIndex,1, firstLayerName, new int[] {1,1},new int[] {1,1},(3*(5+numClasses)), ConvolutionMode.Same, Boolean.FALSE,Boolean.FALSE);

		return secondLayerName;
	}


	/**
	 * Layer 83 => 86
	 * @param graph
	 * @param input
	 * @param skipLayerName
	 * @return
	 */
	private String buildLayerFromEightyThreeToEightySix(ComputationGraphConfiguration.GraphBuilder graph, String input, String skipLayerName) {

		String moduleName="layer-83-to-86";

		int moduleIndex=0;

		input=convBlock(graph, moduleName, moduleIndex,0, input, new int[] {1,1},new int[] {1,1},256, ConvolutionMode.Same, Boolean.TRUE,Boolean.TRUE);

        input=upSampling2D(graph,moduleName,moduleIndex,1,input);

		String thirdLayer=createLayerName(moduleName, MERGE_VERTEX,moduleIndex,2);

		graph.addVertex(thirdLayer, new MergeVertex(), new String[]{input,skipLayerName});

		return thirdLayer;
	}

	private String buildLayerFromEightySevenToNinetyOne(ComputationGraphConfiguration.GraphBuilder graph, String input){

		String moduleName="layer-87-to-91";

		int moduleIndex=0;

		input=convBlock(graph, moduleName, moduleIndex,0, input, new int[] {1,1},new int[] {1,1},256, ConvolutionMode.Same, Boolean.TRUE,Boolean.TRUE);

		input=convBlock(graph, moduleName, moduleIndex,1, input, new int[] {3,3},new int[] {1,1},512, ConvolutionMode.Same, Boolean.TRUE,Boolean.TRUE);

		input=convBlock(graph, moduleName, moduleIndex,2, input, new int[] {1,1},new int[] {1,1},256, ConvolutionMode.Same, Boolean.TRUE,Boolean.TRUE);

		input=convBlock(graph, moduleName, moduleIndex,3, input, new int[] {3,3},new int[] {1,1},512, ConvolutionMode.Same, Boolean.TRUE,Boolean.TRUE);

		input=convBlock(graph, moduleName, moduleIndex,4, input, new int[] {1,1},new int[] {1,1},256, ConvolutionMode.Same, Boolean.TRUE,Boolean.TRUE);

		return input;
	}


	/**
	 * Layer 92 => 94
	 * @param graph
	 * @param input
	 * @return
	 */
	private String buildLayerForMediumObjectDetection(ComputationGraphConfiguration.GraphBuilder graph, String input) {

		String moduleName="layer-92-to-94";

		int moduleIndex=0;

		String firstLayerName=convBlock(graph, moduleName, moduleIndex,0, input, new int[] {3,3},new int[] {1,1},512, ConvolutionMode.Same, Boolean.TRUE,Boolean.TRUE);

		String secondLayerName=convBlock(graph, moduleName, moduleIndex,1, firstLayerName, new int[] {1,1},new int[] {1,1},(3*(5+numClasses)), ConvolutionMode.Same, Boolean.FALSE,Boolean.FALSE);

		return secondLayerName;
	}



	/**
	 * Layer 95 => 98
	 * @param graph
	 * @param input
	 * @param skipLayerName
	 * @return
	 */
	private String buildLayerFromNinetyFiveToNinetyEight(ComputationGraphConfiguration.GraphBuilder graph, String input, String skipLayerName) {

		String moduleName="layer-95-to-98";

		int moduleIndex=0;

		input=convBlock(graph, moduleName, moduleIndex,0, input, new int[] {1,1},new int[] {1,1},128, ConvolutionMode.Same, Boolean.TRUE,Boolean.TRUE);

		input=upSampling2D(graph,moduleName,moduleIndex,1,input);

		String thirdLayer=createLayerName(moduleName, MERGE_VERTEX,moduleIndex,2);

		graph.addVertex(thirdLayer, new MergeVertex(), new String[]{input,skipLayerName});

		return thirdLayer;
	}


	private String buildLayerForBigObjectDetection(ComputationGraphConfiguration.GraphBuilder graph, String input){

		String moduleName="layer-99-to-106";

		int moduleIndex=0;

		input=convBlock(graph, moduleName, moduleIndex,0, input, new int[] {1,1},new int[] {1,1},128, ConvolutionMode.Same, Boolean.TRUE,Boolean.TRUE);

		input=convBlock(graph, moduleName, moduleIndex,1, input, new int[] {3,3},new int[] {1,1},256, ConvolutionMode.Same, Boolean.TRUE,Boolean.TRUE);

		input=convBlock(graph, moduleName, moduleIndex,2, input, new int[] {1,1},new int[] {1,1},128, ConvolutionMode.Same, Boolean.TRUE,Boolean.TRUE);

		input=convBlock(graph, moduleName, moduleIndex,3, input, new int[] {3,3},new int[] {1,1},256, ConvolutionMode.Same, Boolean.TRUE,Boolean.TRUE);

		input=convBlock(graph, moduleName, moduleIndex,4, input, new int[] {1,1},new int[] {1,1},128, ConvolutionMode.Same, Boolean.TRUE,Boolean.TRUE);

		input=convBlock(graph, moduleName, moduleIndex,5, input, new int[] {3,3},new int[] {1,1},256, ConvolutionMode.Same, Boolean.TRUE,Boolean.TRUE);

		input=convBlock(graph, moduleName, moduleIndex,6, input, new int[] {1,1},new int[] {1,1},(3*(5+numClasses)), ConvolutionMode.Same, Boolean.FALSE,Boolean.FALSE);

		return input;
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
