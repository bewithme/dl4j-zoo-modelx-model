package org.freeware.dl4j.modelx.model.backbone;

import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.graph.ElementWiseVertex;
import org.deeplearning4j.nn.conf.layers.*;
import org.freeware.dl4j.nn.GraphLayerItem;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.IUpdater;

import java.util.ArrayList;
import java.util.List;

public class ResNet50Backbone {

    //ResNet50


    public static List<GraphLayerItem> getResNet50Backbone(int nIn, String[] inputs, ConvolutionLayer.AlgoMode aLgoMode, IUpdater updater) {

        List<GraphLayerItem>  graphItemList=new ArrayList<GraphLayerItem>(10);
        // stem
        graphItemList.add(new GraphLayerItem("stem-zero", new ZeroPaddingLayer.Builder(3, 3).build(), inputs));
        graphItemList.add(new GraphLayerItem("stem-cnn1",
                new ConvolutionLayer.Builder(new int[] {7, 7}, new int[] {2, 2}).nOut(64)
                        .build(),
                new String[]{"stem-zero"}));
        graphItemList.add(new GraphLayerItem("stem-batch1", new BatchNormalization(), new String[]{"stem-cnn1"}));
        graphItemList.add(new GraphLayerItem("stem-act1", new ActivationLayer.Builder().activation(Activation.RELU).build(),
                new String[]{"stem-batch1"}));
        graphItemList.add(new GraphLayerItem("stem-maxpool1", new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX,
                new int[] {3, 3}, new int[] {2, 2}).build(), new String[]{"stem-act1"}));


        graphItemList.addAll(convBlock( new int[] {3, 3}, new int[] {64, 64, 256}, "2", "a", new int[] {2, 2}, "stem-maxpool1",aLgoMode,updater));
        graphItemList.addAll(identityBlock( new int[] {3, 3}, new int[] {64, 64, 256}, "2", "b", "res2a_branch",aLgoMode,updater));
        graphItemList.addAll(identityBlock( new int[] {3, 3}, new int[] {64, 64, 256}, "2", "c", "res2b_branch",aLgoMode,updater));

        graphItemList.addAll(convBlock(new int[] {3, 3}, new int[] {128, 128, 512}, "3", "a", "res2c_branch",aLgoMode,updater));
        graphItemList.addAll(identityBlock( new int[] {3, 3}, new int[] {128, 128, 512}, "3", "b", "res3a_branch",aLgoMode,updater));
        graphItemList.addAll(identityBlock( new int[] {3, 3}, new int[] {128, 128, 512}, "3", "c", "res3b_branch",aLgoMode,updater));
        graphItemList.addAll(identityBlock( new int[] {3, 3}, new int[] {128, 128, 512}, "3", "d", "res3c_branch",aLgoMode,updater));

        graphItemList.addAll(convBlock( new int[] {3, 3}, new int[] {256, 256, 1024}, "4", "a", "res3d_branch",aLgoMode,updater));
        graphItemList.addAll(identityBlock( new int[] {3, 3}, new int[] {256, 256, 1024}, "4", "b", "res4a_branch",aLgoMode,updater));
        graphItemList.addAll(identityBlock( new int[] {3, 3}, new int[] {256, 256, 1024}, "4", "c", "res4b_branch",aLgoMode,updater));
        graphItemList.addAll(identityBlock( new int[] {3, 3}, new int[] {256, 256, 1024}, "4", "d", "res4c_branch",aLgoMode,updater));
        graphItemList.addAll(identityBlock( new int[] {3, 3}, new int[] {256, 256, 1024}, "4", "e", "res4d_branch",aLgoMode,updater));
        graphItemList.addAll(identityBlock( new int[] {3, 3}, new int[] {256, 256, 1024}, "4", "f", "res4e_branch",aLgoMode,updater));

        graphItemList.addAll(convBlock( new int[] {3, 3}, new int[] {512, 512, 2048}, "5", "a", "res4f_branch",aLgoMode,updater));
        graphItemList.addAll(identityBlock( new int[] {3, 3}, new int[] {512, 512, 2048}, "5", "b", "res5a_branch",aLgoMode,updater));
        graphItemList.addAll(identityBlock( new int[] {3, 3}, new int[] {512, 512, 2048}, "5", "c", "res5b_branch",aLgoMode,updater));

        graphItemList.add(new GraphLayerItem("avgpool",
                new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[] {3, 3}).build(),
                new String[]{"res5c_branch"}));


        return graphItemList;
    }


    private static List<GraphLayerItem> convBlock( int[] kernelSize, int[] filters,
                                            String stage, String block, String input,ConvolutionLayer.AlgoMode aLgoMode, IUpdater updater) {
        return  convBlock( kernelSize, filters, stage, block, new int[] {2, 2}, input,aLgoMode,updater);
    }
    private static  List<GraphLayerItem> identityBlock(int[] kernelSize, int[] filters, String stage, String block, String input,ConvolutionLayer.AlgoMode aLgoMode, IUpdater updater) {

        String convName = "res" + stage + block + "_branch";
        String batchName = "bn" + stage + block + "_branch";
        String activationName = "act" + stage + block + "_branch";
        String shortcutName = "short" + stage + block + "_branch";

        List<GraphLayerItem>  graphItemList=new ArrayList<GraphLayerItem>(10);

        graphItemList.add(new GraphLayerItem(convName + "2a",
                new ConvolutionLayer.Builder(new int[] {1, 1})
                        .nOut(filters[0])
                        .updater(updater)
                        .cudnnAlgoMode(aLgoMode)
                        .build(),
                new String[]{input}));
        graphItemList.add(new GraphLayerItem(batchName + "2a", new BatchNormalization.Builder()
                .updater(updater).build(), new String[]{convName + "2a"}));
        graphItemList.add(new GraphLayerItem(activationName + "2a",
                new ActivationLayer.Builder().activation(Activation.RELU).build(),
                new String[]{batchName + "2a"}));

        graphItemList.add(new GraphLayerItem(convName + "2b", new ConvolutionLayer.Builder(kernelSize)
                .nOut(filters[1])
                .updater(updater)
                .cudnnAlgoMode(aLgoMode).convolutionMode(ConvolutionMode.Same).build(),
                new String[]{activationName + "2a"}));
        graphItemList.add(new GraphLayerItem(batchName + "2b", new BatchNormalization.Builder()
                .updater(updater)
                .build(), new String[]{convName + "2b"}));
        graphItemList.add(new GraphLayerItem(activationName + "2b",
                new ActivationLayer.Builder().activation(Activation.RELU).build(),
                new String[]{batchName + "2b"}));

        graphItemList.add(new GraphLayerItem(convName + "2c",
                new ConvolutionLayer.Builder(new int[] {1, 1}).nOut(filters[2])
                        .updater(updater)
                        .cudnnAlgoMode(aLgoMode).build(),
                new String[]{activationName + "2b"}));
        graphItemList.add(new GraphLayerItem(batchName + "2c", new BatchNormalization.Builder()
                .updater(updater).build(), new String[]{convName + "2c"}));

        graphItemList.add(new GraphLayerItem(shortcutName, new ElementWiseVertex(ElementWiseVertex.Op.Add),new String[]{ batchName + "2c",
                input}));
        graphItemList.add(new GraphLayerItem(convName, new ActivationLayer.Builder().activation(Activation.RELU).build(),
                new String[]{shortcutName}));

        return graphItemList;
    }

    private static  List<GraphLayerItem>   convBlock(int[] kernelSize, int[] filters, String stage, String block, int[] stride, String input,ConvolutionLayer.AlgoMode aLgoMode, IUpdater updater) {
        String convName = "res" + stage + block + "_branch";
        String batchName = "bn" + stage + block + "_branch";
        String activationName = "act" + stage + block + "_branch";
        String shortcutName = "short" + stage + block + "_branch";

        List<GraphLayerItem>  graphItemList=new ArrayList<GraphLayerItem>(10);

        graphItemList.add(new GraphLayerItem(convName + "2a", new ConvolutionLayer.Builder(new int[] {1, 1}, stride)
                .cudnnAlgoMode(aLgoMode)
                .updater(updater)
                .nOut(filters[0]).build(),
                new String[]{input}));
        graphItemList.add(new GraphLayerItem(batchName + "2a", new BatchNormalization.Builder()
                .updater(updater).build(), new String[]{convName + "2a"}));
        graphItemList.add(new GraphLayerItem(activationName + "2a",
                new ActivationLayer.Builder().activation(Activation.RELU).build(),
                new String[]{batchName + "2a"}));

        graphItemList.add(new GraphLayerItem(convName + "2b",
                new ConvolutionLayer.Builder(kernelSize).nOut(filters[1])
                        .updater(updater)
                        .convolutionMode(ConvolutionMode.Same).build(),
                new String[]{activationName + "2a"}));
        graphItemList.add(new GraphLayerItem(batchName + "2b", new BatchNormalization.Builder()
                .updater(updater).build(), new String[]{convName + "2b"}));
        graphItemList.add(new GraphLayerItem(activationName + "2b",
                new ActivationLayer.Builder().activation(Activation.RELU).build(),
                new String[]{batchName + "2b"}));
        graphItemList.add(new GraphLayerItem(convName + "2c",
                new ConvolutionLayer.Builder(new int[] {1, 1})
                        .cudnnAlgoMode(aLgoMode)
                        .nOut(filters[2])
                        .updater(updater)
                        .build(),
                new String[]{activationName + "2b"}));
        graphItemList.add(new GraphLayerItem(batchName + "2c", new BatchNormalization.Builder()
                .updater(updater).build(), new String[]{convName + "2c"}));

        // shortcut
        graphItemList.add(new GraphLayerItem(convName + "1",
                new ConvolutionLayer.Builder(new int[] {1, 1}, stride)
                        .cudnnAlgoMode(aLgoMode)
                        .nOut(filters[2])
                        .updater(updater)
                        .build(),
                new String[]{input}));
        graphItemList.add(new GraphLayerItem(batchName + "1", new BatchNormalization.Builder()
                .updater(updater).build(), new String[]{convName + "1"}));


        graphItemList.add(new GraphLayerItem(shortcutName,
                new ElementWiseVertex(ElementWiseVertex.Op.Add),
                new String[]{batchName + "2c",batchName + "1"}));

        graphItemList.add(new GraphLayerItem(convName, new ActivationLayer.Builder().activation(Activation.RELU).build(),
                new String[]{shortcutName}));

        return graphItemList;
    }

}
