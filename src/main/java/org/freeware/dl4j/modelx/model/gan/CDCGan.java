package org.freeware.dl4j.modelx.model.gan;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.*;
import org.deeplearning4j.nn.conf.graph.ElementWiseVertex;
import org.deeplearning4j.nn.conf.graph.MergeVertex;
import org.deeplearning4j.nn.conf.graph.ReshapeVertex;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.conf.preprocessor.CnnToFeedForwardPreProcessor;
import org.deeplearning4j.nn.conf.preprocessor.FeedForwardToCnnPreProcessor;
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
public class CDCGan extends AbsGan{

    @Builder.Default private long seed = 12345;

    @Builder.Default private WorkspaceMode workspaceMode = WorkspaceMode.ENABLED;

    private static final double LEARNING_RATE = 0.0003;

    private static final double GRADIENT_THRESHOLD = 100.0;

    private static final IUpdater UPDATER_ZERO = Sgd.builder().learningRate(0.0).build();

    private  static int DISCRIMINATOR_INPUT_SIZE =784;

    private  static int GENERATOR_INPUT_SIZE =100;

    private  static int LATENT_DIM_LEN =100;

    @Builder.Default private   int numClasses =10;

    @Builder.Default private   int imageHeight =28;

    @Builder.Default private   int imageWidth =28;

    @Builder.Default private   int imageChannel =1;

    @Builder.Default  private IUpdater updater = Adam.builder().learningRate(LEARNING_RATE).beta1(0.5).build();


    private static Random random =new Random(123456);


    /**
     * 生成器网络配置
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

        String[] inputs= {"latent_dim","label_num"};

        List<GraphLayerItem>  layerItems=buildGeneratorGraphLayerItems(inputs);

        addGraphItems(graph,layerItems,Boolean.FALSE);

        graph.addInputs(inputs);

        graph.inputPreProcessor("gen_layer_1",new FeedForwardToCnnPreProcessor(7, 7, 256));

        graph.setOutputs("gen_layer_8");

        return   graph.build();
    }

    /**
     * 判别器网络配置
     * @return
     */
    public ComputationGraphConfiguration buildDiscriminatorConfiguration() {

        ComputationGraphConfiguration.GraphBuilder graph = new NeuralNetConfiguration.Builder().seed(seed)

                .updater(updater)
                .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
                .gradientNormalizationThreshold(GRADIENT_THRESHOLD)
                .weightInit(WeightInit.XAVIER)
                .activation(Activation.IDENTITY)
                .trainingWorkspaceMode(workspaceMode)
                .inferenceWorkspaceMode(workspaceMode)
                .convolutionMode(ConvolutionMode.Same)
                .graphBuilder();

        String[] inputs= {"image","label_num"};

        List<GraphLayerItem>   layerItems=buildDiscriminatorGraphLayerItems(inputs, updater);

        addGraphItems(graph,layerItems,Boolean.FALSE);

        graph.addInputs(inputs);

        graph.inputPreProcessor("dis_layer_8", new CnnToFeedForwardPreProcessor(4, 4, 128));

        graph.setOutputs("dis_layer_8");

        return   graph.build();
    }


    public ComputationGraphConfiguration buildGanConfiguration() {

        ComputationGraphConfiguration.GraphBuilder graph = new NeuralNetConfiguration.Builder().seed(seed)

                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(updater)
                .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
                .gradientNormalizationThreshold(GRADIENT_THRESHOLD)
                .weightInit(WeightInit.XAVIER)
                .activation(Activation.IDENTITY)
                .trainingWorkspaceMode(workspaceMode)
                .inferenceWorkspaceMode(workspaceMode)
                .convolutionMode(ConvolutionMode.Same)
                .graphBuilder();

        String[] genInputs= {"latent_dim","label_num"};

        List<GraphLayerItem>  genLayerItems=buildGeneratorGraphLayerItems(genInputs);

        addGraphItems(graph,genLayerItems,Boolean.FALSE);

        String[] disInputs={"gen_layer_8","label_num"};
        //学习率为0，即判别器不会被训练，只训练生成器
        List<GraphLayerItem>  disLayerItems=buildDiscriminatorGraphLayerItems(disInputs,UPDATER_ZERO);

        addGraphItems(graph,disLayerItems,Boolean.FALSE);

        graph.addInputs(genInputs);

        graph.setOutputs("dis_layer_8");

        graph.inputPreProcessor("gen_layer_1",new FeedForwardToCnnPreProcessor(7, 7, 256));

        graph.inputPreProcessor("dis_layer_8", new CnnToFeedForwardPreProcessor(4, 4, 128));

        return   graph.build();
    }


    /**
     * 生成器计算图的层列表
     * @param inputs
     * @return
     */
    private  List<GraphLayerItem> buildGeneratorGraphLayerItems(String[] inputs){

        List<GraphLayerItem>  graphItemList=new ArrayList<GraphLayerItem>(10);

        graphItemList.add(new GraphLayerItem("gen_embedding_0",
                new EmbeddingLayer.Builder()
                        .nIn(numClasses)
                        .nOut(LATENT_DIM_LEN)
                        .build(),
                new String[]{inputs[1]}));

        graphItemList.add(new GraphLayerItem("gen_vertex_0",
                new ElementWiseVertex(ElementWiseVertex.Op.Product),
                new String[]{inputs[0],"gen_embedding_0"}));

        graphItemList.add(new GraphLayerItem("gen_layer_0",
                new DenseLayer.Builder()
                        .nIn(LATENT_DIM_LEN)
                        .nOut(256*7*7)
                        .weightInit(WeightInit.NORMAL)
                        .build(),
                new String[]{"gen_vertex_0"}));

        graphItemList.add(new GraphLayerItem("gen_layer_1",
                new Deconvolution2D.Builder()
                        .nIn(256)
                        .nOut(128)
                        .kernelSize(3,3)
                        .stride(2,2)
                        .convolutionMode(ConvolutionMode.Same)
                        .build(),
                new String[]{"gen_layer_0"}));

        graphItemList.add(new GraphLayerItem("gen_layer_2",
                new BatchNormalization.Builder()
                        .nIn(128)
                        .nOut(128)

                        .build(),
                new String[]{"gen_layer_1"}));

        graphItemList.add(new GraphLayerItem("gen_layer_3",
                new ActivationLayer.Builder(new ActivationLReLU(0.01)).build(),
                new String[]{"gen_layer_2"}));

        graphItemList.add(new GraphLayerItem("gen_layer_4",
                new Deconvolution2D.Builder()
                        .nIn(128)
                        .nOut(64)
                        .kernelSize(3,3)
                        .stride(1,1)
                        .convolutionMode(ConvolutionMode.Same)
                        .build(),
                new String[]{"gen_layer_3"}));

        graphItemList.add(new GraphLayerItem("gen_layer_5",
                new BatchNormalization.Builder()
                        .nIn(64)
                        .nOut(64)
                        .build(),
                new String[]{"gen_layer_4"}));

        graphItemList.add(new GraphLayerItem("gen_layer_6",
                new ActivationLayer.Builder(new ActivationLReLU(0.01)).build(),
                new String[]{"gen_layer_5"}));

        graphItemList.add(new GraphLayerItem("gen_layer_7",
                new Deconvolution2D.Builder()
                        .nIn(64)
                        .nOut(1)
                        .kernelSize(3,3)
                        .stride(2,2)
                        .convolutionMode(ConvolutionMode.Same)
                        .build(),
                new String[]{"gen_layer_6"}));

        graphItemList.add(new GraphLayerItem("gen_layer_8",
                new ActivationLayer.Builder(new ActivationTanH()).build(),
                new String[]{"gen_layer_7"}));

        return graphItemList;

    }

    /**
     * 判别器计算图的层列表
     * @param inputs
     * @param updater
     * @return
     */
    private  List<GraphLayerItem> buildDiscriminatorGraphLayerItems(String[] inputs,IUpdater updater){

        List<GraphLayerItem>  graphItemList=new ArrayList<GraphLayerItem>(10);

        graphItemList.add(new GraphLayerItem("dis_embedding_0",
                new EmbeddingLayer.Builder()
                        .nIn(numClasses)
                        .nOut(imageHeight * imageWidth * imageChannel).build(),
                new String[]{inputs[1]}));

        graphItemList.add(new GraphLayerItem("dis_reshape_0",
                //-1 means keep the same batch size of original array
                new ReshapeVertex(-1, imageChannel, imageHeight, imageWidth),
                new String[]{"dis_embedding_0"}));

        graphItemList.add(new GraphLayerItem("dis_merge_vertex_0",
                new MergeVertex(),
                new String[]{inputs[0],"dis_reshape_0"}));

        graphItemList.add(new GraphLayerItem("dis_layer_0",
                new Convolution2D.Builder()
                        .nIn(imageChannel+imageChannel)
                        .kernelSize(3,3)
                        .stride(2,2)
                        .nOut(64)
                        .updater(updater).build(),
                new String[]{"dis_merge_vertex_0"}));

        graphItemList.add(new GraphLayerItem("dis_layer_1",
                new ActivationLayer.Builder(new ActivationLReLU(0.01))
                        .build(),
                new String[]{"dis_layer_0"}));

        graphItemList.add(new GraphLayerItem("dis_layer_2",
                new Convolution2D.Builder()
                        .nIn(64)
                        .kernelSize(3,3)
                        .stride(2,2)
                        .nOut(64)
                        .updater(updater).build(),
                new String[]{"dis_layer_1"}));

        graphItemList.add(new GraphLayerItem("dis_layer_3",
                new BatchNormalization.Builder()
                        .nIn(64)
                        .nOut(64)
                        .updater(updater)
                        .build(),
                new String[]{"dis_layer_2"}));

        graphItemList.add(new GraphLayerItem("dis_layer_4",
                new ActivationLayer.Builder(new ActivationLReLU(0.01)).build(),
                new String[]{"dis_layer_3"}));

        graphItemList.add(new GraphLayerItem("dis_layer_5",
                new Convolution2D.Builder()
                        .nIn(64)
                        .kernelSize(3,3)
                        .stride(2,2)
                        .nOut(128)
                        .updater(updater).build(),
                new String[]{"dis_layer_4"}));


        graphItemList.add(new GraphLayerItem("dis_layer_6",
                new BatchNormalization.Builder()
                        .nIn(128)
                        .nOut(128)
                        .updater(updater)
                        .build(),
                new String[]{"dis_layer_5"}));

        graphItemList.add(new GraphLayerItem("dis_layer_7",
                new ActivationLayer.Builder(new ActivationLReLU(0.01)).build(),
                new String[]{"dis_layer_6"}));

        graphItemList.add(new GraphLayerItem("dis_layer_8",
                new OutputLayer.Builder(LossFunctions.LossFunction.XENT)
                        .nIn(2048)
                        .nOut(1)
                        .activation(Activation.SIGMOID)
                        .updater(updater).build(),
                new String[]{"dis_layer_7"}));

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
