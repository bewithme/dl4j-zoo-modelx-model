package org.freeware.dl4j.modelx.model.gan;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.conf.distribution.TruncatedNormalDistribution;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.preprocessor.CnnToFeedForwardPreProcessor;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.IWeightInit;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.nn.weights.WeightInitDistribution;
import org.deeplearning4j.zoo.ModelMetaData;
import org.deeplearning4j.zoo.PretrainedType;
import org.freeware.dl4j.modelx.model.backbone.ResNet50Backbone;
import org.freeware.dl4j.modelx.model.backbone.UNetBackbone;
import org.freeware.dl4j.nn.GraphLayerItem;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.learning.config.IUpdater;
import org.nd4j.linalg.learning.config.Sgd;

import java.util.List;
import java.util.Random;


@AllArgsConstructor
@Builder
@Slf4j
public class InPaintingGanV2 extends AbsGan{

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



    public ComputationGraphConfiguration buildDiscriminatorConfiguration() {
        IWeightInit weightInit = new WeightInitDistribution(new TruncatedNormalDistribution(0.0, 0.5));
        ComputationGraphConfiguration.GraphBuilder graph = new NeuralNetConfiguration.Builder().seed(seed)
                .activation(Activation.IDENTITY)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(discriminatorUpdater)
                .weightInit(weightInit)
                .l1(1e-7)
                .l2(5e-5)
                .miniBatch(true)
                .trainingWorkspaceMode(workspaceMode)
                .inferenceWorkspaceMode(workspaceMode)
                .cudnnAlgoMode(cudnnAlgoMode)
                .convolutionMode(ConvolutionMode.Truncate)
                .graphBuilder();

        String[] inputs= {"dis_input"};

        ResNet50Backbone.setBackbone(graph,inputs,discriminatorUpdater);

        ResNet50Backbone.setCnnLossLayer(graph,discriminatorUpdater);

        graph.addInputs(inputs);

        String lastLayerName="output-layer";

        //graph.inputPreProcessor(lastLayerName, new CnnToFeedForwardPreProcessor(3, 3, 2048));

        graph.setOutputs(lastLayerName);

        graph.setInputTypes(InputType.convolutional(imageHeight, imageWidth, imageChannel));

        return   graph.build();
    }


    public ComputationGraphConfiguration buildGanConfiguration() {


        ComputationGraphConfiguration.GraphBuilder graph = new NeuralNetConfiguration.Builder().seed(seed)
                .activation(Activation.IDENTITY)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(discriminatorUpdater)
                .weightInit(WeightInit.XAVIER)
                .l1(1e-7)
                .l2(5e-5)
                .miniBatch(true)
                .trainingWorkspaceMode(workspaceMode)
                .inferenceWorkspaceMode(workspaceMode)
                .cudnnAlgoMode(cudnnAlgoMode)
                .convolutionMode(ConvolutionMode.Truncate)
                .graphBuilder();

        String[] genInputs= {"gen_input"};

        List<GraphLayerItem>  genLayerItems=buildGeneratorGraphLayerItems(genInputs);

        addGraphItems(graph,genLayerItems,Boolean.FALSE);

        String[] disInputs={"conv10"};

        graph.addInputs(genInputs);

        ResNet50Backbone.setBackbone(graph,disInputs,UPDATER_ZERO);

        ResNet50Backbone.setCnnLossLayer(graph,UPDATER_ZERO);

        String lastLayerName="output-layer";

        //graph.inputPreProcessor(lastLayerName, new CnnToFeedForwardPreProcessor(3, 3, 2048));

        graph.setOutputs(lastLayerName);

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
