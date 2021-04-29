package org.freeware.dl4j.modelx.model.backbone;

import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.graph.ElementWiseVertex;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.freeware.dl4j.nn.GraphLayerItem;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.IUpdater;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.ArrayList;
import java.util.List;

public class ResNet50Backbone {

    private static ConvolutionLayer.AlgoMode cudnnAlgoMode = ConvolutionLayer.AlgoMode.PREFER_FASTEST;


    public static void setBackbone(ComputationGraphConfiguration.GraphBuilder graph,String[] inputs,IUpdater updater){

                graph
                // stem
                .addLayer("stem-zero", new ZeroPaddingLayer.Builder(3, 3).build(), inputs)
                .addLayer("stem-cnn1",
                        new ConvolutionLayer.Builder(new int[] {7, 7}, new int[] {2, 2})
                                .convolutionMode(ConvolutionMode.Truncate)
                                .updater(updater)
                                .nOut(64)
                                .build(),
                        "stem-zero")
                .addLayer("stem-batch1", new BatchNormalization.Builder().updater(updater)
                        .build(), "stem-cnn1")
                .addLayer("stem-act1", new ActivationLayer.Builder().activation(Activation.RELU).build(),
                        "stem-batch1")
                .addLayer("stem-maxpool1", new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX,
                        new int[] {3, 3}, new int[] {2, 2})
                        .convolutionMode(ConvolutionMode.Truncate)
                        .build(), "stem-act1");

        convBlock(graph, new int[] {3, 3}, new int[] {64, 64, 256}, "2", "a", new int[] {2, 2}, "stem-maxpool1",updater);
        identityBlock(graph, new int[] {3, 3}, new int[] {64, 64, 256}, "2", "b", "res2a_branch",updater);
        identityBlock(graph, new int[] {3, 3}, new int[] {64, 64, 256}, "2", "c", "res2b_branch",updater);

        convBlock(graph, new int[] {3, 3}, new int[] {128, 128, 512}, "3", "a", "res2c_branch",updater);
        identityBlock(graph, new int[] {3, 3}, new int[] {128, 128, 512}, "3", "b", "res3a_branch",updater);
        identityBlock(graph, new int[] {3, 3}, new int[] {128, 128, 512}, "3", "c", "res3b_branch",updater);
        identityBlock(graph, new int[] {3, 3}, new int[] {128, 128, 512}, "3", "d", "res3c_branch",updater);

        convBlock(graph, new int[] {3, 3}, new int[] {256, 256, 1024}, "4", "a", "res3d_branch",updater);
        identityBlock(graph, new int[] {3, 3}, new int[] {256, 256, 1024}, "4", "b", "res4a_branch",updater);
        identityBlock(graph, new int[] {3, 3}, new int[] {256, 256, 1024}, "4", "c", "res4b_branch",updater);
        identityBlock(graph, new int[] {3, 3}, new int[] {256, 256, 1024}, "4", "d", "res4c_branch",updater);
        identityBlock(graph, new int[] {3, 3}, new int[] {256, 256, 1024}, "4", "e", "res4d_branch",updater);
        identityBlock(graph, new int[] {3, 3}, new int[] {256, 256, 1024}, "4", "f", "res4e_branch",updater);

        convBlock(graph, new int[] {3, 3}, new int[] {512, 512, 2048}, "5", "a", "res4f_branch",updater);
        identityBlock(graph, new int[] {3, 3}, new int[] {512, 512, 2048}, "5", "b", "res5a_branch",updater);
        identityBlock(graph, new int[] {3, 3}, new int[] {512, 512, 2048}, "5", "c", "res5b_branch",updater);

        graph.addLayer("avgpool",
                new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[] {3, 3}).build(),
                "res5c_branch");

    }


    public static  void setOutputLayer(ComputationGraphConfiguration.GraphBuilder graph,IUpdater updater){

        graph.addLayer("output-layer", new OutputLayer.Builder(LossFunctions.LossFunction.XENT)
                .updater(updater)
                .nOut(1)
                .activation(Activation.SIGMOID)
                .updater(updater).build(),"avgpool");
    }


    private static void identityBlock(ComputationGraphConfiguration.GraphBuilder graph, int[] kernelSize, int[] filters,
                               String stage, String block, String input,IUpdater updater) {
        String convName = "res" + stage + block + "_branch";
        String batchName = "bn" + stage + block + "_branch";
        String activationName = "act" + stage + block + "_branch";
        String shortcutName = "short" + stage + block + "_branch";

        graph.addLayer(convName + "2a",
                new ConvolutionLayer.Builder(new int[] {1, 1})
                        .nOut(filters[0])
                        .convolutionMode(ConvolutionMode.Truncate)
                        .cudnnAlgoMode(cudnnAlgoMode)
                        .updater(updater)
                        .build(),
                input)
                .addLayer(batchName + "2a", new BatchNormalization.Builder().updater(updater)
                        .build(), convName + "2a")
                .addLayer(activationName + "2a",
                        new ActivationLayer.Builder().activation(Activation.RELU).build(),
                        batchName + "2a")

                .addLayer(convName + "2b", new ConvolutionLayer.Builder(kernelSize)
                                .nOut(filters[1])
                                .updater(updater)
                                .cudnnAlgoMode(cudnnAlgoMode)
                                .convolutionMode(ConvolutionMode.Same)
                                .build(),
                        activationName + "2a")
                .addLayer(batchName + "2b", new BatchNormalization.Builder().updater(updater)
                        .build(), convName + "2b")
                .addLayer(activationName + "2b",
                        new ActivationLayer.Builder().activation(Activation.RELU).build(),
                        batchName + "2b")

                .addLayer(convName + "2c",
                        new ConvolutionLayer.Builder(new int[] {1, 1})
                                .convolutionMode(ConvolutionMode.Truncate)
                                .nOut(filters[2])
                                .updater(updater)
                                .cudnnAlgoMode(cudnnAlgoMode).build(),
                        activationName + "2b")
                .addLayer(batchName + "2c", new BatchNormalization.Builder().updater(updater)
                        .build(), convName + "2c")

                .addVertex(shortcutName, new ElementWiseVertex(ElementWiseVertex.Op.Add), batchName + "2c",
                        input)
                .addLayer(convName, new ActivationLayer.Builder().activation(Activation.RELU).build(),
                        shortcutName);
    }

    private static void convBlock(ComputationGraphConfiguration.GraphBuilder graph, int[] kernelSize, int[] filters, String stage, String block, String input,IUpdater updater) {

        convBlock(graph, kernelSize, filters, stage, block, new int[] {2, 2}, input,updater);
    }

    private static void convBlock(ComputationGraphConfiguration.GraphBuilder graph, int[] kernelSize, int[] filters, String stage, String block, int[] stride, String input,IUpdater updater) {

        String convName = "res" + stage + block + "_branch";

        String batchName = "bn" + stage + block + "_branch";

        String activationName = "act" + stage + block + "_branch";

        String shortcutName = "short" + stage + block + "_branch";

        graph.addLayer(convName + "2a", new ConvolutionLayer.Builder(new int[] {1, 1}, stride)
                        .convolutionMode(ConvolutionMode.Truncate)
                        .updater(updater)
                        .nOut(filters[0])
                        .build(),
                input)
                .addLayer(batchName + "2a", new BatchNormalization.Builder().updater(updater)
                        .build(), convName + "2a")
                .addLayer(activationName + "2a",
                        new ActivationLayer.Builder().activation(Activation.RELU).build(),
                        batchName + "2a")

                .addLayer(convName + "2b",
                        new ConvolutionLayer.Builder(kernelSize)
                                .nOut(filters[1])
                                .updater(updater)
                                .convolutionMode(ConvolutionMode.Same)
                                .build(),
                        activationName + "2a")
                .addLayer(batchName + "2b", new BatchNormalization.Builder().updater(updater)
                        .build(), convName + "2b")
                .addLayer(activationName + "2b",
                        new ActivationLayer.Builder().activation(Activation.RELU).build(),
                        batchName + "2b")

                .addLayer(convName + "2c",
                        new ConvolutionLayer.Builder(new int[] {1, 1})
                                .convolutionMode(ConvolutionMode.Truncate)
                                .updater(updater)
                                .nOut(filters[2])
                                .build(),
                        activationName + "2b")
                .addLayer(batchName + "2c", new BatchNormalization.Builder().updater(updater)
                        .build(), convName + "2c")

                // shortcut
                .addLayer(convName + "1",
                        new ConvolutionLayer.Builder(new int[] {1, 1}, stride)
                                .convolutionMode(ConvolutionMode.Truncate)
                                .updater(updater)
                                .nOut(filters[2])
                                .build(),
                        input)
                .addLayer(batchName + "1", new BatchNormalization.Builder()
                        .updater(updater)
                        .build(), convName + "1")


                .addVertex(shortcutName, new ElementWiseVertex(ElementWiseVertex.Op.Add), batchName + "2c",
                        batchName + "1")
                .addLayer(convName, new ActivationLayer.Builder().activation(Activation.RELU).build(),
                        shortcutName);
    }

}
