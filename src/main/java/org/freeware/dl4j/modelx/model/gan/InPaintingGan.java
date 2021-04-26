package org.freeware.dl4j.modelx.model.gan;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.*;
import org.deeplearning4j.nn.conf.graph.MergeVertex;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.conf.preprocessor.CnnToFeedForwardPreProcessor;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.zoo.ModelMetaData;
import org.deeplearning4j.zoo.PretrainedType;
import org.freeware.dl4j.nn.GraphLayerItem;
import org.nd4j.linalg.activations.Activation;
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
public class InPaintingGan extends AbsGan{

    @Builder.Default private long seed = 12345;

    @Builder.Default private WorkspaceMode workspaceMode = WorkspaceMode.ENABLED;

    private static final double LEARNING_RATE = 0.0003;

    private static final double GRADIENT_THRESHOLD = 100.0;

    private static final IUpdater UPDATER_ZERO = Sgd.builder().learningRate(0.0).build();

    @Builder.Default private   int imageHeight =512;

    @Builder.Default private   int imageWidth =512;

    @Builder.Default private   int imageChannel =3;

    @Builder.Default  private IUpdater generatorUpdater = Adam.builder().learningRate(LEARNING_RATE).beta1(0.5).build();

    @Builder.Default  private IUpdater discriminatorUpdater = Adam.builder().learningRate(LEARNING_RATE).beta1(0.5).build();

    @Builder.Default private ConvolutionLayer.AlgoMode cudnnAlgoMode = ConvolutionLayer.AlgoMode.PREFER_FASTEST;

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

        String[] inputs= {"gen_input"};

        List<GraphLayerItem>  layerItems=buildGeneratorGraphLayerItems(inputs);

        addGraphItems(graph,layerItems,Boolean.FALSE);

        graph.addInputs(inputs);

        graph.setOutputs("conv10");

        graph.setInputTypes(InputType.convolutional(imageHeight, imageWidth, imageChannel));

        return   graph.build();
    }

    /**
     * 判别器网络配置
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

        String[] inputs= {"dis_input"};

        List<GraphLayerItem>   layerItems=buildDiscriminatorGraphLayerItems(inputs, generatorUpdater);

        addGraphItems(graph,layerItems,Boolean.FALSE);

        graph.addInputs(inputs);

        graph.inputPreProcessor("dis_layer_10", new CnnToFeedForwardPreProcessor(32, 32, 1));

        graph.setOutputs("dis_layer_10");

        return   graph.build();
    }


    public ComputationGraphConfiguration buildGanConfiguration() {

        ComputationGraphConfiguration.GraphBuilder graph = new NeuralNetConfiguration.Builder().seed(seed)

                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(generatorUpdater)
                //.l2(5e-5)
                //.gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
               // .gradientNormalizationThreshold(GRADIENT_THRESHOLD)
                .weightInit(WeightInit.RELU)
                //.activation(Activation.IDENTITY)
                .trainingWorkspaceMode(workspaceMode)
                .inferenceWorkspaceMode(workspaceMode)
                .convolutionMode(ConvolutionMode.Same)
                .graphBuilder();

        String[] genInputs= {"gen_input"};

        List<GraphLayerItem>  genLayerItems=buildGeneratorGraphLayerItems(genInputs);

        addGraphItems(graph,genLayerItems,Boolean.FALSE);

        String[] disInputs={"conv10"};
        //学习率为0，即判别器不会被训练，只训练生成器
        List<GraphLayerItem>  disLayerItems=buildDiscriminatorGraphLayerItems(disInputs,UPDATER_ZERO);

        addGraphItems(graph,disLayerItems,Boolean.FALSE);

        graph.addInputs(genInputs);

        graph.setOutputs("dis_layer_10");

        graph.inputPreProcessor("dis_layer_10", new CnnToFeedForwardPreProcessor(32, 32, 1));


        graph.setInputTypes(InputType.convolutional(imageHeight, imageWidth, imageChannel));

        return   graph.build();
    }


    /**
     * 生成器计算图的层列表
     * UNet作为backbone
     * @param inputs
     * @return
     */
    private  List<GraphLayerItem> buildGeneratorGraphLayerItems(String[] inputs){

        List<GraphLayerItem>  graphItemList=new ArrayList<GraphLayerItem>(10);

        graphItemList.add(new GraphLayerItem(
                "conv1-1", new ConvolutionLayer.Builder(3,3)
                .stride(1,1)
                .nIn(imageChannel)
                .nOut(64)
                .convolutionMode(ConvolutionMode.Same).cudnnAlgoMode(cudnnAlgoMode)
                .activation(Activation.RELU).build(),
                inputs));


        graphItemList.add(new GraphLayerItem(
                "conv1-2", new ConvolutionLayer.Builder(3,3).stride(1,1).nOut(64)
                .convolutionMode(ConvolutionMode.Same).cudnnAlgoMode(cudnnAlgoMode)
                .activation(Activation.RELU).build(),
                new String[]{"conv1-1"}));


        graphItemList.add(new GraphLayerItem(
                "pool1", new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX).kernelSize(2,2)
                .build(),
                new String[]{"conv1-2"}));

        graphItemList.add(new GraphLayerItem("conv2-1", new ConvolutionLayer.Builder(3,3).stride(1,1).nOut(128)
                .convolutionMode(ConvolutionMode.Same).cudnnAlgoMode(cudnnAlgoMode)
                .activation(Activation.RELU).build(), new String[]{"pool1"}));

        graphItemList.add(new GraphLayerItem("conv2-2", new ConvolutionLayer.Builder(3,3).stride(1,1).nOut(128)
                        .convolutionMode(ConvolutionMode.Same).cudnnAlgoMode(cudnnAlgoMode)
                        .activation(Activation.RELU).build(), new String[]{"conv2-1"}));
        graphItemList.add(new GraphLayerItem("pool2", new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX).kernelSize(2,2)
                        .build(), new String[]{"conv2-2"}));

        graphItemList.add(new GraphLayerItem("conv3-1", new ConvolutionLayer.Builder(3,3).stride(1,1).nOut(256)
                        .convolutionMode(ConvolutionMode.Same).cudnnAlgoMode(cudnnAlgoMode)
                        .activation(Activation.RELU).build(), new String[]{"pool2"}));
        graphItemList.add(new GraphLayerItem("conv3-2", new ConvolutionLayer.Builder(3,3).stride(1,1).nOut(256)
                        .convolutionMode(ConvolutionMode.Same).cudnnAlgoMode(cudnnAlgoMode)
                        .activation(Activation.RELU).build(), new String[]{"conv3-1"}));
        graphItemList.add(new GraphLayerItem("pool3", new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX).kernelSize(2,2)
                        .build(), new String[]{"conv3-2"}));
        graphItemList.add(new GraphLayerItem("conv4-1", new ConvolutionLayer.Builder(3,3).stride(1,1).nOut(512)
                .convolutionMode(ConvolutionMode.Same).cudnnAlgoMode(cudnnAlgoMode)
                .activation(Activation.RELU).build(), new String[]{"pool3"}));
        graphItemList.add(new GraphLayerItem("conv4-2", new ConvolutionLayer.Builder(3,3).stride(1,1).nOut(512)
                .convolutionMode(ConvolutionMode.Same).cudnnAlgoMode(cudnnAlgoMode)
                .activation(Activation.RELU).build(), new String[]{"conv4-1"}));
        graphItemList.add(new GraphLayerItem("drop4", new DropoutLayer.Builder(0.5).build(), new String[]{"conv4-2"}));

        graphItemList.add(new GraphLayerItem("pool4", new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX).kernelSize(2,2)
                .build(), new String[]{"drop4"}));

        graphItemList.add(new GraphLayerItem("conv5-1", new ConvolutionLayer.Builder(3,3).stride(1,1).nOut(1024)
                .convolutionMode(ConvolutionMode.Same).cudnnAlgoMode(cudnnAlgoMode)
                .activation(Activation.RELU).build(), new String[]{"pool4"}));
        graphItemList.add(new GraphLayerItem("conv5-2", new ConvolutionLayer.Builder(3,3).stride(1,1).nOut(1024)
                .convolutionMode(ConvolutionMode.Same).cudnnAlgoMode(cudnnAlgoMode)
                .activation(Activation.RELU).build(), new String[]{"conv5-1"}));
        graphItemList.add(new GraphLayerItem("drop5", new DropoutLayer.Builder(0.5).build(), new String[]{"conv5-2"}));

        // up6
        graphItemList.add(new GraphLayerItem("up6-1", new Upsampling2D.Builder(2).build(), new String[]{"drop5"}));
        graphItemList.add(new GraphLayerItem("up6-2", new ConvolutionLayer.Builder(2,2)
                .stride(1,1)
                .nOut(512)
                .convolutionMode(ConvolutionMode.Same).cudnnAlgoMode(cudnnAlgoMode)
                .activation(Activation.RELU).build(), new String[]{"up6-1"}));
        graphItemList.add(new GraphLayerItem("merge6", new MergeVertex(), new String[]{"drop4", "up6-2"}));
        graphItemList.add(new GraphLayerItem("conv6-1", new ConvolutionLayer.Builder(3,3)
                .stride(1,1)
                .nOut(512)
                .convolutionMode(ConvolutionMode.Same).cudnnAlgoMode(cudnnAlgoMode)
                .activation(Activation.RELU).build(), new String[]{"merge6"}));
        graphItemList.add(new GraphLayerItem("conv6-2", new ConvolutionLayer.Builder(3,3)
                .stride(1,1)
                .nOut(512)
                .convolutionMode(ConvolutionMode.Same).cudnnAlgoMode(cudnnAlgoMode)
                .activation(Activation.RELU).build(), new String[]{"conv6-1"}));

        // up7
        graphItemList.add(new GraphLayerItem("up7-1", new Upsampling2D.Builder(2).build(), new String[]{"conv6-2"}));
        graphItemList.add(new GraphLayerItem("up7-2", new ConvolutionLayer.Builder(2,2)
                .stride(1,1)
                .nOut(256)
                .convolutionMode(ConvolutionMode.Same).cudnnAlgoMode(cudnnAlgoMode)
                .activation(Activation.RELU).build(), new String[]{"up7-1"}));
        graphItemList.add(new GraphLayerItem("merge7", new MergeVertex(), new String[]{"conv3-2", "up7-2"}));
        graphItemList.add(new GraphLayerItem("conv7-1", new ConvolutionLayer.Builder(3,3)
                .stride(1,1)
                .nOut(256)
                .convolutionMode(ConvolutionMode.Same).cudnnAlgoMode(cudnnAlgoMode)
                .activation(Activation.RELU).build(), new String[]{"merge7"}));
        graphItemList.add(new GraphLayerItem("conv7-2", new ConvolutionLayer.Builder(3,3)
                .stride(1,1)
                .nOut(256)
                .convolutionMode(ConvolutionMode.Same).cudnnAlgoMode(cudnnAlgoMode)
                .activation(Activation.RELU).build(), new String[]{"conv7-1"}));

        // up8
        graphItemList.add(new GraphLayerItem("up8-1", new Upsampling2D.Builder(2).build(), new String[]{"conv7-2"}));
        graphItemList.add(new GraphLayerItem("up8-2", new ConvolutionLayer.Builder(2,2)
                .stride(1,1)
                .nOut(128)
                .convolutionMode(ConvolutionMode.Same).cudnnAlgoMode(cudnnAlgoMode)
                .activation(Activation.RELU).build(), new String[]{"up8-1"}));
        graphItemList.add(new GraphLayerItem("merge8", new MergeVertex(),new String[]{ "conv2-2", "up8-2"}));
        graphItemList.add(new GraphLayerItem("conv8-1", new ConvolutionLayer.Builder(3,3)
                .stride(1,1)
                .nOut(128)
                .convolutionMode(ConvolutionMode.Same).cudnnAlgoMode(cudnnAlgoMode)
                .activation(Activation.RELU).build(), new String[]{"merge8"}));
        graphItemList.add(new GraphLayerItem("conv8-2", new ConvolutionLayer.Builder(3,3)
                .stride(1,1)
                .nOut(128)
                .convolutionMode(ConvolutionMode.Same).cudnnAlgoMode(cudnnAlgoMode)
                .activation(Activation.RELU).build(), new String[]{"conv8-1"}));

        // up9
        graphItemList.add(new GraphLayerItem("up9-1", new Upsampling2D.Builder(2).build(), new String[]{"conv8-2"}));
        graphItemList.add(new GraphLayerItem("up9-2", new ConvolutionLayer.Builder(2,2)
                .stride(1,1)
                .nOut(64)
                .convolutionMode(ConvolutionMode.Same).cudnnAlgoMode(cudnnAlgoMode)
                .activation(Activation.RELU).build(), new String[]{"up9-1"}));
        graphItemList.add(new GraphLayerItem("merge9", new MergeVertex(), new String[]{"conv1-2", "up9-2"}));
        graphItemList.add(new GraphLayerItem("conv9-1", new ConvolutionLayer.Builder(3,3)
                .stride(1,1)
                .nOut(64)
                .convolutionMode(ConvolutionMode.Same)
                .cudnnAlgoMode(cudnnAlgoMode)
                .activation(Activation.RELU).build(), new String[]{"merge9"}));
        graphItemList.add(new GraphLayerItem("conv9-2", new ConvolutionLayer.Builder(3,3)
                .stride(1,1)
                .nOut(64)
                .convolutionMode(ConvolutionMode.Same)
                .cudnnAlgoMode(cudnnAlgoMode)
                .activation(Activation.RELU).build(), new String[]{"conv9-1"}));
        graphItemList.add(new GraphLayerItem("conv9-3", new ConvolutionLayer.Builder(3,3)
                .stride(1,1)
                .nOut(2)
                .convolutionMode(ConvolutionMode.Same)
                .cudnnAlgoMode(cudnnAlgoMode)
                .activation(Activation.RELU).build(), new String[]{"conv9-2"}));

        graphItemList.add(new GraphLayerItem("conv10", new ConvolutionLayer.Builder(1,1)
                .stride(1,1).nOut(3)
                .convolutionMode(ConvolutionMode.Same)
                .cudnnAlgoMode(cudnnAlgoMode)
                .activation(Activation.IDENTITY).build(), new String[]{"conv9-3"}));

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

        graphItemList.add(new GraphLayerItem("dis_layer_0", new ConvolutionLayer.Builder(new int[]{4,4}, new int[]{2,2})
                .cudnnAlgoMode(cudnnAlgoMode).convolutionMode(ConvolutionMode.Same)
                .activation(Activation.LEAKYRELU)
                .updater(updater)
                .nIn(imageChannel)
                .nOut(64)
                .build(), inputs));

        graphItemList.add(new GraphLayerItem("dis_layer_1", new ConvolutionLayer.Builder(new int[]{4,4}, new int[]{2,2})
                .cudnnAlgoMode(cudnnAlgoMode).convolutionMode(ConvolutionMode.Same)
                .updater(updater)
                .activation(Activation.LEAKYRELU)
                .nIn(64)
                .nOut(128)
                .build(),new String[]{"dis_layer_0"}));

        graphItemList.add(new GraphLayerItem("dis_layer_2", new BatchNormalization.Builder()
                .updater(updater)
                .nIn(128)
                .nOut(128)
                .build(),new String[]{"dis_layer_1"}));

        graphItemList.add(new GraphLayerItem("dis_layer_3", new ConvolutionLayer.Builder(new int[]{4,4}, new int[]{2,2})
                .cudnnAlgoMode(cudnnAlgoMode)
                .convolutionMode(ConvolutionMode.Same)
                .updater(updater)
                .activation(Activation.LEAKYRELU)
                .nIn(128)
                .nOut(256)
                .build(),new String[]{"dis_layer_2"}));

        graphItemList.add(new GraphLayerItem("dis_layer_4", new BatchNormalization.Builder()
                .updater(updater)
                .nIn(256)
                .nOut(256)
                .build(),new String[]{"dis_layer_3"}));

        graphItemList.add(new GraphLayerItem("dis_layer_5", new ConvolutionLayer.Builder(new int[]{4,4}, new int[]{2,2})
                .cudnnAlgoMode(cudnnAlgoMode)
                .convolutionMode(ConvolutionMode.Same)
                .updater(updater)
                .activation(Activation.LEAKYRELU)
                .nIn(256)
                .nOut(512)
                .build(),new String[]{"dis_layer_4"}));

        graphItemList.add(new GraphLayerItem("dis_layer_6", new BatchNormalization.Builder()
                .updater(updater)
                .nIn(512)
                .nOut(512)
                .build(),new String[]{"dis_layer_5"}));

        graphItemList.add(new GraphLayerItem("dis_layer_7", new ConvolutionLayer.Builder(new int[]{4,4})
                .cudnnAlgoMode(cudnnAlgoMode)
                .convolutionMode(ConvolutionMode.Same)
                .updater(updater)
                .nIn(512)
                .nOut(512)
                .activation(Activation.LEAKYRELU)
                .build(),new String[]{"dis_layer_6"}));

        graphItemList.add(new GraphLayerItem("dis_layer_8", new BatchNormalization.Builder()
                .updater(updater)
                .nIn(512)
                .nOut(512)
                .build(),new String[]{"dis_layer_7"}));

        graphItemList.add(new GraphLayerItem("dis_layer_9", new ConvolutionLayer.Builder(new int[]{4,4})
                .cudnnAlgoMode(cudnnAlgoMode)
                .convolutionMode(ConvolutionMode.Same)
                .updater(updater)
                .nIn(512)
                .nOut(1)
                .build(),new String[]{"dis_layer_8"}));


        graphItemList.add(new GraphLayerItem("dis_layer_10",
                new OutputLayer.Builder(LossFunctions.LossFunction.XENT)
                        .updater(updater)
                        .nIn(1024)
                        .nOut(1)
                        .activation(Activation.SIGMOID)
                        .updater(updater).build(),
                new String[]{"dis_layer_9"}));

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
