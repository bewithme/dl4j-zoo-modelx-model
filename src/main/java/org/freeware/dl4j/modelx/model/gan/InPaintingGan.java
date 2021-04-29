package org.freeware.dl4j.modelx.model.gan;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.*;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.graph.MergeVertex;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.conf.preprocessor.CnnToFeedForwardPreProcessor;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.zoo.ModelMetaData;
import org.deeplearning4j.zoo.PretrainedType;
import org.freeware.dl4j.modelx.model.backbone.ResNet50Backbone;
import org.freeware.dl4j.modelx.model.backbone.UNetBackbone;
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
     *
     * @return
     */
    public ComputationGraphConfiguration buildGeneratorConfiguration() {

        ComputationGraphConfiguration.GraphBuilder graph = new NeuralNetConfiguration.Builder().seed(seed)

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
     * ???????
     * @return
     */
    public ComputationGraphConfiguration buildDiscriminatorConfiguration() {

        ComputationGraphConfiguration.GraphBuilder graph = new NeuralNetConfiguration.Builder().seed(seed)

                .updater(discriminatorUpdater)
                .weightInit(new NormalDistribution(0.0, 0.02))
                .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
                .gradientNormalizationThreshold(GRADIENT_THRESHOLD)
                .trainingWorkspaceMode(workspaceMode)
                .inferenceWorkspaceMode(workspaceMode)
                .convolutionMode(ConvolutionMode.Same)
                .graphBuilder();

        String[] inputs= {"dis_input"};

        List<GraphLayerItem>   layerItems=buildDiscriminatorGraphLayerItems(inputs, generatorUpdater);

        addGraphItems(graph,layerItems,Boolean.FALSE);

        graph.addInputs(inputs);

        String lastLayerName=getLastLayerName(layerItems);

        graph.inputPreProcessor(lastLayerName, new CnnToFeedForwardPreProcessor(32, 32, 1));

        graph.setOutputs(lastLayerName);

        return   graph.build();
    }


    public ComputationGraphConfiguration buildGanConfiguration() {

        ComputationGraphConfiguration.GraphBuilder graph = new NeuralNetConfiguration.Builder().seed(seed)

                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(generatorUpdater)
                //.l2(5e-5)
                //.gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
                // .gradientNormalizationThreshold(GRADIENT_THRESHOLD)
                //.weightInit(WeightInit.RELU)
                //.activation(Activation.IDENTITY)
                .trainingWorkspaceMode(workspaceMode)
                .inferenceWorkspaceMode(workspaceMode)
                .convolutionMode(ConvolutionMode.Same)
                .graphBuilder();

        String[] genInputs= {"gen_input"};

        List<GraphLayerItem>  genLayerItems=buildGeneratorGraphLayerItems(genInputs);

        addGraphItems(graph,genLayerItems,Boolean.FALSE);

        String[] disInputs={"conv10"};
        //????0?????????????????
        List<GraphLayerItem>  disLayerItems=buildDiscriminatorGraphLayerItems(disInputs,UPDATER_ZERO);

        addGraphItems(graph,disLayerItems,Boolean.FALSE);

        graph.addInputs(genInputs);

        graph.setOutputs("dis_layer_10");

        graph.inputPreProcessor("dis_layer_10", new CnnToFeedForwardPreProcessor(32, 32, 1));


        graph.setInputTypes(InputType.convolutional(imageHeight, imageWidth, imageChannel));

        return   graph.build();
    }


    /**
     * ??????????
     * UNet??backbone
     * @param inputs
     * @return
     */
    private  List<GraphLayerItem> buildGeneratorGraphLayerItems(String[] inputs){

        List<GraphLayerItem>  graphItemList= UNetBackbone.getUNetBackbone(imageChannel,inputs,cudnnAlgoMode);

        return graphItemList;

    }

    /**
     * ??????????
     * @param inputs
     * @param updater
     * @return
     */
    private  List<GraphLayerItem> buildDiscriminatorGraphLayerItems(String[] inputs,IUpdater updater){


        return ResNet50Backbone.getResNet50Backbone(imageChannel,inputs,cudnnAlgoMode,updater);

        /**

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

         **/

    }


    @Override
    public void setInputShape(int[][] inputShape) {

    }


    /**
     * ???????
     * @return
     */
    public ComputationGraph init() {

        ComputationGraphConfiguration configuration=buildGanConfiguration();

        ComputationGraph gan = new ComputationGraph(configuration);

        gan.init();

        return gan;

    }

    /**
     * ??????
     * @return
     */
    public ComputationGraph initGenerator() {

        ComputationGraphConfiguration genConf=buildGeneratorConfiguration();

        ComputationGraph model = new ComputationGraph(genConf);

        model.init();

        return model;
    }

    /**
     * ??????
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
