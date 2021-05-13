package org.freeware.dl4j.modelx.model.gan;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.*;
import org.deeplearning4j.nn.conf.graph.ElementWiseVertex;

import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;

import org.deeplearning4j.nn.graph.ComputationGraph;

import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.zoo.ModelMetaData;
import org.deeplearning4j.zoo.PretrainedType;

import org.freeware.dl4j.nn.GraphLayerItem;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.activations.impl.ActivationLReLU;
import org.nd4j.linalg.activations.impl.ActivationTanH;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.learning.config.IUpdater;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;




@AllArgsConstructor
@Builder
@Slf4j
public class SRGan extends AbsGan {

    @Builder.Default
    private long seed = 12345;

    @Builder.Default
    private WorkspaceMode workspaceMode = WorkspaceMode.ENABLED;

    private static final double LEARNING_RATE = 0.0003;

    private static final double GRADIENT_THRESHOLD = 100.0;

    private static final IUpdater UPDATER_ZERO = Sgd.builder().learningRate(0.0).build();

    @Builder.Default
    private int imageHeight = 64;

    @Builder.Default
    private int imageWidth = 64;

    @Builder.Default
    private int imageChannel = 3;

    @Builder.Default
    private IUpdater generatorUpdater = Adam.builder().learningRate(LEARNING_RATE).beta1(0.5).build();

    @Builder.Default
    private IUpdater discriminatorUpdater = Adam.builder().learningRate(LEARNING_RATE).beta1(0.5).build();

    @Builder.Default
    private ConvolutionLayer.AlgoMode cudnnAlgoMode = ConvolutionLayer.AlgoMode.PREFER_FASTEST;

    private static Random random = new Random(123456);


    private static int generatorFilters = 32;

    private static int discriminatorFilters = 64;


    private ComputationGraph vgg19=null;

    public ComputationGraph getVgg19() {
        return vgg19;
    }

    /**
     * 生成器网络配置
     *
     * @return
     */
    public ComputationGraphConfiguration buildGeneratorConfiguration() {

        ComputationGraphConfiguration.GraphBuilder graph = new NeuralNetConfiguration.Builder().seed(seed)
                .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
                .gradientNormalizationThreshold(GRADIENT_THRESHOLD)
                .weightInit(WeightInit.XAVIER)
                .activation(Activation.IDENTITY)
                .trainingWorkspaceMode(workspaceMode)
                .inferenceWorkspaceMode(workspaceMode)
                .convolutionMode(ConvolutionMode.Same)
                .graphBuilder();

        String inputs = "gen-input";

        String moduleName = "gen";

        List<GraphLayerItem> layerItems = buildGeneratorGraphLayerItems(inputs,moduleName,generatorUpdater);

        addGraphItems(graph, layerItems, Boolean.FALSE);

        graph.addInputs(inputs);

        graph.setOutputs(getLastLayerName(layerItems));

        graph.setInputTypes(InputType.convolutional(imageHeight, imageWidth, imageChannel));

        return graph.build();
    }

    /**
     * 判别器网络配置
     *
     * @return
     */
    public ComputationGraphConfiguration buildDiscriminatorConfiguration() {

        ComputationGraphConfiguration.GraphBuilder graph = new NeuralNetConfiguration.Builder().seed(seed)

                .updater(discriminatorUpdater)
                .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
                .gradientNormalizationThreshold(GRADIENT_THRESHOLD)
                .weightInit(WeightInit.XAVIER)
                .activation(Activation.IDENTITY)
                .trainingWorkspaceMode(workspaceMode)
                .inferenceWorkspaceMode(workspaceMode)
                .convolutionMode(ConvolutionMode.Same)
                .graphBuilder();

        String input ="dis-input";

        List<GraphLayerItem> layerItems = buildDiscriminatorGraphLayerItems(input, discriminatorUpdater);

        addGraphItems(graph, layerItems, Boolean.FALSE);

        graph.addInputs(input);

        graph.setOutputs(getLastLayerName(layerItems));

        graph.setInputTypes(InputType.convolutional(256, 256, 3));

        return graph.build();
    }


    public ComputationGraphConfiguration buildGanConfiguration() {



        ComputationGraphConfiguration.GraphBuilder graph = new NeuralNetConfiguration.Builder().seed(seed)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .trainingWorkspaceMode(workspaceMode)
                .inferenceWorkspaceMode(workspaceMode)
                .convolutionMode(ConvolutionMode.Same)
                .graphBuilder();

        String genInputs = "gen-input";

        String moduleName = "gen";

        List<GraphLayerItem> genLayerItems = buildGeneratorGraphLayerItems(genInputs,moduleName,generatorUpdater);

        addGraphItems(graph, genLayerItems, Boolean.FALSE);

        String disInput = getLastLayerName(genLayerItems);
        //学习率为0，即判别器不会被训练，只训练生成器
        List<GraphLayerItem> disLayerItems = buildDiscriminatorGraphLayerItems(disInput, UPDATER_ZERO);

        addGraphItems(graph, disLayerItems, Boolean.FALSE);

        graph.addInputs(genInputs);


        graph.addLayer("cnn-loss-output",new CnnLossLayer.Builder(LossFunctions.LossFunction.MSE)
                .updater(UPDATER_ZERO)
                .activation(Activation.TANH).build(),disInput);

        graph.setOutputs(getLastLayerName(disLayerItems),"cnn-loss-output");

        graph.setInputTypes(InputType.convolutional(imageHeight, imageWidth, imageChannel));


        return graph.build();
    }






    /**
     * 生成器计算图的层列表
     *
     *
     * @param input
     * @return
     */
    private List<GraphLayerItem> buildGeneratorGraphLayerItems(String input, String moduleName,IUpdater updater) {

        double gamma=0.8;

        List<GraphLayerItem> graphItemList = new ArrayList<GraphLayerItem>(10);

        String genLayer0=createLayerName(moduleName, CNN,0,0);

        graphItemList.add(new GraphLayerItem(genLayer0,
                new Convolution2D.Builder()
                        .nIn(imageChannel)
                        .kernelSize(9,9)
                        .stride(1,1)
                        .activation(Activation.LEAKYRELU)
                        .nOut(64)
                        .convolutionMode(ConvolutionMode.Same)
                        .updater(updater).build(),
                new String[]{input}));

        List<GraphLayerItem> block0= residualBlock("gen-res-block",1,0,genLayer0,64,updater);

        graphItemList.addAll(block0);

        String inputLayerName=getLastLayerName(block0);

        for(int i=0;i<15;i++){

            List<GraphLayerItem> block= residualBlock("gen-res-block",2,i,inputLayerName,64,updater);

            inputLayerName=getLastLayerName(block);

            graphItemList.addAll(block);
        }

        String genLayer1=createLayerName(moduleName, CNN,3,0);

        graphItemList.add(new GraphLayerItem(genLayer1,
                new Convolution2D.Builder()
                        .nIn(64)
                        .kernelSize(3,3)
                        .stride(1,1)
                        .nOut(64)
                        .convolutionMode(ConvolutionMode.Same)
                        .updater(updater).build(),
                new String[]{inputLayerName}));


        String genLayer2=createLayerName(moduleName, BATCH_NORM,3,1);

        graphItemList.add(new GraphLayerItem(genLayer2,
                new BatchNormalization.Builder()
                        .gamma(gamma)
                        .nIn(64)
                        .nOut(64)
                        .updater(updater)
                        .build(),
                new String[]{genLayer1}));

        String genLayer3=createLayerName(moduleName, ELEMENT_WISE_VERTEX,3,2);

        graphItemList.add(new GraphLayerItem(genLayer3,
                new ElementWiseVertex(ElementWiseVertex.Op.Add),
                new String[]{genLayer0,genLayer2}));


        String genLayer4=createLayerName(moduleName, UP_SAMPLING_2D,4,0);

        graphItemList.add(new GraphLayerItem(genLayer4,
                new Upsampling2D.Builder(2).build(),
                new String[]{genLayer3}));

        String genLayer5=createLayerName(moduleName, CNN,4,1);

        graphItemList.add(new GraphLayerItem(genLayer5,
                new Convolution2D.Builder()
                        .nIn(64)
                        .kernelSize(3,3)
                        .stride(1,1)
                        .nOut(256)
                        .convolutionMode(ConvolutionMode.Same)
                        .updater(updater).build(),
                new String[]{genLayer4}));

        String genLayer6=createLayerName(moduleName, ACTIVATION,4,2);

        graphItemList.add(new GraphLayerItem(genLayer6,
                new ActivationLayer.Builder(new ActivationLReLU()).build(),
                new String[]{genLayer5}));


        //-------------

        String genLayer7=createLayerName(moduleName, UP_SAMPLING_2D,5,0);

        graphItemList.add(new GraphLayerItem(genLayer7,
                new Upsampling2D.Builder(2).build(),
                new String[]{genLayer6}));

        String genLayer8=createLayerName(moduleName, CNN,5,1);

        graphItemList.add(new GraphLayerItem(genLayer8,
                new Convolution2D.Builder()
                        .kernelSize(3,3)
                        .stride(1,1)
                        .nOut(256)
                        .convolutionMode(ConvolutionMode.Same)
                        .updater(updater).build(),
                new String[]{genLayer7}));

        String genLayer9=createLayerName(moduleName, ACTIVATION,5,2);

        graphItemList.add(new GraphLayerItem(genLayer9,
                new ActivationLayer.Builder(new ActivationLReLU()).build(),
                new String[]{genLayer8}));


        //-------------
        String genLayer10=createLayerName(moduleName, CNN,6,0);

        graphItemList.add(new GraphLayerItem(genLayer10,
                new Convolution2D.Builder()
                        .kernelSize(9,9)
                        .stride(1,1)
                        .nOut(imageChannel)
                        .convolutionMode(ConvolutionMode.Same)
                        .updater(updater).build(),
                new String[]{genLayer9}));

        String genLayer11=createLayerName(moduleName, ACTIVATION,6,1);

        graphItemList.add(new GraphLayerItem(genLayer11,
                new ActivationLayer.Builder(new ActivationTanH()).build(),
                new String[]{genLayer10}));


        return graphItemList;

    }

    /**
     * 判别器计算图的层列表
     *
     * @param input
     * @param updater
     * @return
     */
    private List<GraphLayerItem> buildDiscriminatorGraphLayerItems(String input, IUpdater updater) {

        List<GraphLayerItem> graphItemList = new ArrayList<GraphLayerItem>(10);

        String moduleName="dis-con-block";

        String inputLayerName;

        List<GraphLayerItem> list0=convolution2D(moduleName,0,0,input,imageChannel,64,3,1,Boolean.FALSE,updater);

        inputLayerName=getLastLayerName(list0);

        List<GraphLayerItem> list1=convolution2D(moduleName,0,1,inputLayerName,64,64,3,2,Boolean.TRUE,updater);

        inputLayerName=getLastLayerName(list1);
        List<GraphLayerItem> list2=convolution2D(moduleName,0,2,inputLayerName,64,128,3,1,Boolean.TRUE,updater);

        inputLayerName=getLastLayerName(list2);
        List<GraphLayerItem> list3=convolution2D(moduleName,0,3,inputLayerName,128,128,3,2,Boolean.TRUE,updater);

        inputLayerName=getLastLayerName(list3);
        List<GraphLayerItem> list4=convolution2D(moduleName,0,4,inputLayerName,128,256,3,1,Boolean.TRUE,updater);


        inputLayerName=getLastLayerName(list4);
        List<GraphLayerItem> list5=convolution2D(moduleName,0,5,inputLayerName,256,256,3,2,Boolean.TRUE,updater);


        inputLayerName=getLastLayerName(list5);
        List<GraphLayerItem> list6=convolution2D(moduleName,0,6,inputLayerName,256,512,3,1,Boolean.TRUE,updater);

        inputLayerName=getLastLayerName(list6);
        List<GraphLayerItem> list7=convolution2D(moduleName,0,7,inputLayerName,512,512,3,2,Boolean.TRUE,updater);

        inputLayerName=getLastLayerName(list7);

        graphItemList.addAll(list0);
        graphItemList.addAll(list1);
        graphItemList.addAll(list2);
        graphItemList.addAll(list3);
        graphItemList.addAll(list4);
        graphItemList.addAll(list5);
        graphItemList.addAll(list6);
        graphItemList.addAll(list7);

        moduleName="dis-output";

        String denseLayer0=createLayerName(moduleName,DENSE_LAYER,0,0);

        graphItemList.add(new GraphLayerItem(denseLayer0,
                new DenseLayer.Builder()
                        .nIn(131072)
                        .nOut(1024)
                        .build(),
                new String[]{inputLayerName}));


        String actLayer0=createLayerName(moduleName,ACTIVATION,0,1);

        graphItemList.add(new GraphLayerItem(actLayer0,
                new ActivationLayer.Builder(new ActivationLReLU(0.2)).build(),
                new String[]{denseLayer0}));

        String outputLayer0=createLayerName(moduleName,OUTPUT_LAYER,0,2);

        graphItemList.add(new GraphLayerItem(outputLayer0,
                new OutputLayer.Builder(LossFunctions.LossFunction.XENT)
                        .nIn(1024)
                        .nOut(1)
                        .activation(Activation.SIGMOID)
                        .updater(updater).build(),
                new String[]{actLayer0}));


        return graphItemList;

    }



    private  List<GraphLayerItem> residualBlock(String moduleName, int moduleIndex, int blockIndex, String layerInputName, int nIn, IUpdater updater){

        double gamma=0.8;

        List<GraphLayerItem>  graphItemList=new ArrayList<GraphLayerItem>(10);

        String cnnLayerName=createLayerName(moduleName, CNN,moduleIndex,blockIndex);

        graphItemList.add(new GraphLayerItem(cnnLayerName,
                new Convolution2D.Builder()
                        .nIn(nIn)
                        .kernelSize(3,3)
                        .stride(1,1)
                        .nOut(64)
                        .convolutionMode(ConvolutionMode.Same)
                        .updater(updater).build(),
                new String[]{layerInputName}));

        String activationLayerName=createLayerName(moduleName, ACTIVATION,moduleIndex,blockIndex);

        graphItemList.add(new GraphLayerItem(activationLayerName,
                new ActivationLayer.Builder(new ActivationLReLU()).build(),
                new String[]{cnnLayerName}));

        String batchNormLayerName=createLayerName(moduleName, BATCH_NORM,moduleIndex,blockIndex);

        graphItemList.add(new GraphLayerItem(batchNormLayerName,
                    new BatchNormalization.Builder().gamma(gamma)
                            .nIn(64)
                            .nOut(64)
                            .updater(updater)
                            .build(),
                    new String[]{activationLayerName}));


        String cnnLayerName2=createLayerName(moduleName, CNN.concat("-1"),moduleIndex,blockIndex);

        graphItemList.add(new GraphLayerItem(cnnLayerName2,
                new Convolution2D.Builder()
                        .nIn(64)
                        .kernelSize(3,3)
                        .stride(1,1)
                        .nOut(64)
                        .convolutionMode(ConvolutionMode.Same)
                        .updater(updater).build(),
                new String[]{batchNormLayerName}));


        String batchNormLayerName2=createLayerName(moduleName, BATCH_NORM.concat("-1"),moduleIndex,blockIndex);

        graphItemList.add(new GraphLayerItem(batchNormLayerName2,
                new BatchNormalization.Builder()
                        .gamma(gamma)
                        .nIn(64)
                        .nOut(64)
                        .updater(updater)
                        .build(),
                new String[]{cnnLayerName2}));

        String vertexLayerName=createLayerName(moduleName, ELEMENT_WISE_VERTEX,moduleIndex,blockIndex);

        graphItemList.add(new GraphLayerItem(vertexLayerName,
                new ElementWiseVertex(ElementWiseVertex.Op.Add),
                new String[]{batchNormLayerName2,layerInputName}));

        return graphItemList;

    }


    private  List<GraphLayerItem> convolution2D(String moduleName,int moduleIndex,int blockIndex,String layerInputName,int nIn,int nOut,int kernelSize,int stridesSize,boolean normalization,IUpdater updater){

        double   leakyReLuAlpha = 0.2;

        List<GraphLayerItem>  graphItemList=new ArrayList<GraphLayerItem>(10);

        String cnnLayerName=createLayerName(moduleName, CNN,moduleIndex,blockIndex);

        graphItemList.add(new GraphLayerItem(cnnLayerName,
                new Convolution2D.Builder()
                        .nIn(nIn)
                        .kernelSize(kernelSize,kernelSize)
                        .stride(stridesSize,stridesSize)
                        .nOut(nOut)
                        .convolutionMode(ConvolutionMode.Same)
                        .updater(updater).build(),
                new String[]{layerInputName}));

        String activationLayerName=createLayerName(moduleName, ACTIVATION,moduleIndex,blockIndex);

        graphItemList.add(new GraphLayerItem(activationLayerName,
                new ActivationLayer.Builder(new ActivationLReLU(leakyReLuAlpha)).build(),
                new String[]{cnnLayerName}));


        if(normalization){

            String batchNormLayerName=createLayerName(moduleName, BATCH_NORM,moduleIndex,blockIndex);

            graphItemList.add(new GraphLayerItem(batchNormLayerName,
                    new BatchNormalization.Builder()
                            .nIn(nOut)
                            .nOut(nOut)
                            .updater(updater)
                            .build(),
                    new String[]{activationLayerName}));

        }

        return graphItemList;

    }






    @Override
    public void setInputShape(int[][] inputShape) {

    }


    /**
     * 初始化对抗网络
     * @return
     */
    public ComputationGraph init() {

        ComputationGraphConfiguration configuration=buildGanConfiguration();

        ComputationGraph gan = new ComputationGraph(configuration);

        gan.init();

        return gan;

    }

    /**
     * 初始化生成器
     * @return
     */
    public ComputationGraph initGenerator() {

        ComputationGraphConfiguration genConf=buildGeneratorConfiguration();

        ComputationGraph model = new ComputationGraph(genConf);

        model.init();

        return model;
    }

    /**
     * 初始化判别器
     * @return
     */
    public ComputationGraph initDiscriminator() {

        ComputationGraphConfiguration configuration=buildDiscriminatorConfiguration();

        ComputationGraph model = new ComputationGraph(configuration);

        model.init();

        return model;
    }





    @Override
    public ModelMetaData metaData() {
        return null;
    }

    @Override
    public Class<? extends Model> modelType() {
        return null;
    }

    @Override
    public String pretrainedUrl(PretrainedType pretrainedType) {
        return null;
    }

    @Override
    public long pretrainedChecksum(PretrainedType pretrainedType) {
        return 0;
    }
}
