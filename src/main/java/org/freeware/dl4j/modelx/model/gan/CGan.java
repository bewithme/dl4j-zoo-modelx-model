package org.freeware.dl4j.modelx.model.gan;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.*;
import org.deeplearning4j.nn.conf.graph.MergeVertex;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.zoo.ModelMetaData;
import org.deeplearning4j.zoo.PretrainedType;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.activations.impl.ActivationLReLU;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.learning.config.IUpdater;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import javax.swing.*;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;


@AllArgsConstructor
@Builder
@Slf4j
public class CGan extends AbsGan{

    @Builder.Default private long seed = 12345;

    @Builder.Default private WorkspaceMode workspaceMode = WorkspaceMode.ENABLED;

    private static final double LEARNING_RATE = 0.0002;

    private static final double GRADIENT_THRESHOLD = 100.0;

    private static final IUpdater UPDATER = Adam.builder().learningRate(LEARNING_RATE).beta1(0.5).build();

    private static final IUpdater UPDATER_ZERO = Sgd.builder().learningRate(0.0).build();

    private  static int DISCRIMINATOR_INPUT_SIZE =784;

    private  static int GENERATOR_INPUT_SIZE =100;

    private  static int LATENT_DIM_LEN =100;

    private  static int LABEL_NUM =10;



    private static Random random =new Random(123456);


    public ComputationGraphConfiguration buildGeneratorConfiguration() {

        ComputationGraphConfiguration.GraphBuilder graph = new NeuralNetConfiguration.Builder().seed(seed)
                .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
                .gradientNormalizationThreshold(GRADIENT_THRESHOLD)
                .weightInit(WeightInit.XAVIER)
                .activation(Activation.IDENTITY)
                .trainingWorkspaceMode(workspaceMode)
                .inferenceWorkspaceMode(workspaceMode)
                .convolutionMode(ConvolutionMode.Truncate)
                .graphBuilder();

        String[] inputs= {"latent_dim","label_num"};

        List<GraphLayerItem>  layerItems=buildGeneratorGraphLayerItems(inputs);

        addGraphItems(graph,layerItems,Boolean.FALSE);

        graph.addInputs(inputs);

        graph.setOutputs("gen_layer_6");

        return   graph.build();
    }

    public ComputationGraphConfiguration buildDiscriminatorConfiguration() {

        ComputationGraphConfiguration.GraphBuilder graph = new NeuralNetConfiguration.Builder().seed(seed)

                .updater(UPDATER)
                .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
                .gradientNormalizationThreshold(GRADIENT_THRESHOLD)
                .weightInit(WeightInit.XAVIER)
                .activation(Activation.IDENTITY)
                .trainingWorkspaceMode(workspaceMode)
                .inferenceWorkspaceMode(workspaceMode)
                .convolutionMode(ConvolutionMode.Truncate)
                .graphBuilder();

        String[] inputs= {"image","label_num"};

        List<GraphLayerItem>   layerItems=buildDiscriminatorGraphLayerItems(inputs,UPDATER);

        addGraphItems(graph,layerItems,Boolean.FALSE);

        graph.addInputs(inputs);

        graph.setOutputs("dis_layer_9");

        return   graph.build();
    }


    public ComputationGraphConfiguration buildGanConfiguration() {

        ComputationGraphConfiguration.GraphBuilder graph = new NeuralNetConfiguration.Builder().seed(seed)

                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(UPDATER)
                .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
                .gradientNormalizationThreshold(GRADIENT_THRESHOLD)
                .weightInit(WeightInit.XAVIER)
                .activation(Activation.IDENTITY)
                .trainingWorkspaceMode(workspaceMode)
                .inferenceWorkspaceMode(workspaceMode)
                .convolutionMode(ConvolutionMode.Truncate)
                .graphBuilder();

        String[] genInputs= {"latent_dim","label_num"};

        List<GraphLayerItem>  genLayerItems=buildGeneratorGraphLayerItems(genInputs);

        addGraphItems(graph,genLayerItems,Boolean.FALSE);

        String[] disInputs={"gen_layer_6","label_num"};

        List<GraphLayerItem>  disLayerItems=buildDiscriminatorGraphLayerItems(disInputs,UPDATER_ZERO);

        addGraphItems(graph,disLayerItems,Boolean.FALSE);

        graph.addInputs(genInputs);

        graph.setOutputs("dis_layer_9");

        return   graph.build();
    }


    private static List<GraphLayerItem> buildGeneratorGraphLayerItems(String[] inputs){

        List<GraphLayerItem>  graphItemList=new ArrayList<GraphLayerItem>(10);

        graphItemList.add(new GraphLayerItem("gen_merge_vertex_0",
                new MergeVertex(),
                inputs));

        graphItemList.add(new GraphLayerItem("gen_layer_0",
                new DenseLayer.Builder().nIn(LATENT_DIM_LEN+LABEL_NUM).nOut(256).weightInit(WeightInit.NORMAL).build(),
                new String[]{"gen_merge_vertex_0"}));

        graphItemList.add(new GraphLayerItem("gen_layer_1",
                new ActivationLayer.Builder(new ActivationLReLU(0.2)).build(),
                new String[]{"gen_layer_0"}));

        graphItemList.add(new GraphLayerItem("gen_layer_2",
                new DenseLayer.Builder().nIn(256).nOut(512).build(),
                new String[]{"gen_layer_1"}));

        graphItemList.add(new GraphLayerItem("gen_layer_3",
                new ActivationLayer.Builder(new ActivationLReLU(0.2)).build(),
                new String[]{"gen_layer_2"}));

        graphItemList.add(new GraphLayerItem("gen_layer_4",
                new DenseLayer.Builder().nIn(512).nOut(1024).build(),
                new String[]{"gen_layer_3"}));

        graphItemList.add(new GraphLayerItem("gen_layer_5",
                new ActivationLayer.Builder(new ActivationLReLU(0.2)).build(),
                new String[]{"gen_layer_4"}));

        graphItemList.add(new GraphLayerItem("gen_layer_6",
                new DenseLayer.Builder().nIn(1024).nOut(DISCRIMINATOR_INPUT_SIZE).activation(Activation.TANH).build(),
                new String[]{"gen_layer_5"}));

        return graphItemList;

    }

    private static List<GraphLayerItem> buildDiscriminatorGraphLayerItems(String[] inputs,IUpdater updater){

        List<GraphLayerItem>  graphItemList=new ArrayList<GraphLayerItem>(10);

        graphItemList.add(new GraphLayerItem("dis_merge_vertex_0",
                new MergeVertex(),
                inputs));

        graphItemList.add(new GraphLayerItem("dis_layer_0",
                new DenseLayer.Builder().nIn(DISCRIMINATOR_INPUT_SIZE+LABEL_NUM).nOut(1024).updater(updater).build(),
                new String[]{"dis_merge_vertex_0"}));

        graphItemList.add(new GraphLayerItem("dis_layer_1",
                new ActivationLayer.Builder(new ActivationLReLU(0.2)).build(),
                new String[]{"dis_layer_0"}));

        graphItemList.add(new GraphLayerItem("dis_layer_2",
                new DropoutLayer.Builder(1 - 0.5).build(),
                new String[]{"dis_layer_1"}));

        graphItemList.add(new GraphLayerItem("dis_layer_3",
                new DenseLayer.Builder().nIn(1024).nOut(512).updater(updater).build(),
                new String[]{"dis_layer_2"}));

        graphItemList.add(new GraphLayerItem("dis_layer_4",
                new ActivationLayer.Builder(new ActivationLReLU(0.2)).build(),
                new String[]{"dis_layer_3"}));

        graphItemList.add(new GraphLayerItem("dis_layer_5",
                new DropoutLayer.Builder(1 - 0.5).build(),
                new String[]{"dis_layer_4"}));

        graphItemList.add(new GraphLayerItem("dis_layer_6",
                new DenseLayer.Builder().nIn(512).nOut(256).updater(updater).build(),
                new String[]{"dis_layer_5"}));


        graphItemList.add(new GraphLayerItem("dis_layer_7",
                new ActivationLayer.Builder(new ActivationLReLU(0.2)).build(),
                new String[]{"dis_layer_6"}));

        graphItemList.add(new GraphLayerItem("dis_layer_8",
                new DropoutLayer.Builder(1 - 0.5).build(),
                new String[]{"dis_layer_7"}));

        graphItemList.add(new GraphLayerItem("dis_layer_9",
                new OutputLayer.Builder(LossFunctions.LossFunction.XENT).nIn(256).nOut(1).activation(Activation.SIGMOID).updater(updater).build(),
                new String[]{"dis_layer_8"}));

        return graphItemList;

    }


    @Override
    public void setInputShape(int[][] inputShape) {

    }

    @Override
    public ComputationGraph init() {

        ComputationGraphConfiguration configuration=buildGanConfiguration();

        ComputationGraph model = new ComputationGraph(configuration);

        model.init();

        return model;

    }

    public ComputationGraph initGenerator() {

        ComputationGraphConfiguration genConf=buildGeneratorConfiguration();

        ComputationGraph model = new ComputationGraph(genConf);

        model.init();

        return model;
    }

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
