package org.freeware.dl4j.modelx.model.backbone;

import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.graph.MergeVertex;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DropoutLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.conf.layers.Upsampling2D;
import org.freeware.dl4j.nn.GraphLayerItem;
import org.nd4j.linalg.activations.Activation;

import java.util.ArrayList;
import java.util.List;

public class UNetBackbone {

    public static List<GraphLayerItem> getUNetBackbone(int nIn, String[] inputs, ConvolutionLayer.AlgoMode aLgoMode){

        List<GraphLayerItem>  graphItemList=new ArrayList<GraphLayerItem>(10);

        graphItemList.add(new GraphLayerItem(
                "conv1-1", new ConvolutionLayer.Builder(3,3)
                .stride(1,1)
                .nIn(nIn)
                .nOut(64)
                .convolutionMode(ConvolutionMode.Same).cudnnAlgoMode(aLgoMode)
                .activation(Activation.RELU).build(),
                inputs));


        graphItemList.add(new GraphLayerItem(
                "conv1-2", new ConvolutionLayer.Builder(3,3).stride(1,1).nOut(64)
                .convolutionMode(ConvolutionMode.Same).cudnnAlgoMode(aLgoMode)
                .activation(Activation.RELU).build(),
                new String[]{"conv1-1"}));


        graphItemList.add(new GraphLayerItem(
                "pool1", new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX).kernelSize(2,2)
                .build(),
                new String[]{"conv1-2"}));

        graphItemList.add(new GraphLayerItem("conv2-1", new ConvolutionLayer.Builder(3,3).stride(1,1).nOut(128)
                .convolutionMode(ConvolutionMode.Same).cudnnAlgoMode(aLgoMode)
                .activation(Activation.RELU).build(), new String[]{"pool1"}));

        graphItemList.add(new GraphLayerItem("conv2-2", new ConvolutionLayer.Builder(3,3).stride(1,1).nOut(128)
                .convolutionMode(ConvolutionMode.Same).cudnnAlgoMode(aLgoMode)
                .activation(Activation.RELU).build(), new String[]{"conv2-1"}));
        graphItemList.add(new GraphLayerItem("pool2", new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX).kernelSize(2,2)
                .build(), new String[]{"conv2-2"}));

        graphItemList.add(new GraphLayerItem("conv3-1", new ConvolutionLayer.Builder(3,3).stride(1,1).nOut(256)
                .convolutionMode(ConvolutionMode.Same).cudnnAlgoMode(aLgoMode)
                .activation(Activation.RELU).build(), new String[]{"pool2"}));
        graphItemList.add(new GraphLayerItem("conv3-2", new ConvolutionLayer.Builder(3,3).stride(1,1).nOut(256)
                .convolutionMode(ConvolutionMode.Same).cudnnAlgoMode(aLgoMode)
                .activation(Activation.RELU).build(), new String[]{"conv3-1"}));
        graphItemList.add(new GraphLayerItem("pool3", new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX).kernelSize(2,2)
                .build(), new String[]{"conv3-2"}));
        graphItemList.add(new GraphLayerItem("conv4-1", new ConvolutionLayer.Builder(3,3).stride(1,1).nOut(512)
                .convolutionMode(ConvolutionMode.Same).cudnnAlgoMode(aLgoMode)
                .activation(Activation.RELU).build(), new String[]{"pool3"}));
        graphItemList.add(new GraphLayerItem("conv4-2", new ConvolutionLayer.Builder(3,3).stride(1,1).nOut(512)
                .convolutionMode(ConvolutionMode.Same).cudnnAlgoMode(aLgoMode)
                .activation(Activation.RELU).build(), new String[]{"conv4-1"}));
        graphItemList.add(new GraphLayerItem("drop4", new DropoutLayer.Builder(0.5).build(), new String[]{"conv4-2"}));

        graphItemList.add(new GraphLayerItem("pool4", new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX).kernelSize(2,2)
                .build(), new String[]{"drop4"}));

        graphItemList.add(new GraphLayerItem("conv5-1", new ConvolutionLayer.Builder(3,3).stride(1,1).nOut(1024)
                .convolutionMode(ConvolutionMode.Same).cudnnAlgoMode(aLgoMode)
                .activation(Activation.RELU).build(), new String[]{"pool4"}));
        graphItemList.add(new GraphLayerItem("conv5-2", new ConvolutionLayer.Builder(3,3).stride(1,1).nOut(1024)
                .convolutionMode(ConvolutionMode.Same).cudnnAlgoMode(aLgoMode)
                .activation(Activation.RELU).build(), new String[]{"conv5-1"}));
        graphItemList.add(new GraphLayerItem("drop5", new DropoutLayer.Builder(0.5).build(), new String[]{"conv5-2"}));

        // up6
        graphItemList.add(new GraphLayerItem("up6-1", new Upsampling2D.Builder(2).build(), new String[]{"drop5"}));
        graphItemList.add(new GraphLayerItem("up6-2", new ConvolutionLayer.Builder(2,2)
                .stride(1,1)
                .nOut(512)
                .convolutionMode(ConvolutionMode.Same).cudnnAlgoMode(aLgoMode)
                .activation(Activation.RELU).build(), new String[]{"up6-1"}));
        graphItemList.add(new GraphLayerItem("merge6", new MergeVertex(), new String[]{"drop4", "up6-2"}));
        graphItemList.add(new GraphLayerItem("conv6-1", new ConvolutionLayer.Builder(3,3)
                .stride(1,1)
                .nOut(512)
                .convolutionMode(ConvolutionMode.Same).cudnnAlgoMode(aLgoMode)
                .activation(Activation.RELU).build(), new String[]{"merge6"}));
        graphItemList.add(new GraphLayerItem("conv6-2", new ConvolutionLayer.Builder(3,3)
                .stride(1,1)
                .nOut(512)
                .convolutionMode(ConvolutionMode.Same).cudnnAlgoMode(aLgoMode)
                .activation(Activation.RELU).build(), new String[]{"conv6-1"}));

        // up7
        graphItemList.add(new GraphLayerItem("up7-1", new Upsampling2D.Builder(2).build(), new String[]{"conv6-2"}));
        graphItemList.add(new GraphLayerItem("up7-2", new ConvolutionLayer.Builder(2,2)
                .stride(1,1)
                .nOut(256)
                .convolutionMode(ConvolutionMode.Same).cudnnAlgoMode(aLgoMode)
                .activation(Activation.RELU).build(), new String[]{"up7-1"}));
        graphItemList.add(new GraphLayerItem("merge7", new MergeVertex(), new String[]{"conv3-2", "up7-2"}));
        graphItemList.add(new GraphLayerItem("conv7-1", new ConvolutionLayer.Builder(3,3)
                .stride(1,1)
                .nOut(256)
                .convolutionMode(ConvolutionMode.Same).cudnnAlgoMode(aLgoMode)
                .activation(Activation.RELU).build(), new String[]{"merge7"}));
        graphItemList.add(new GraphLayerItem("conv7-2", new ConvolutionLayer.Builder(3,3)
                .stride(1,1)
                .nOut(256)
                .convolutionMode(ConvolutionMode.Same).cudnnAlgoMode(aLgoMode)
                .activation(Activation.RELU).build(), new String[]{"conv7-1"}));

        // up8
        graphItemList.add(new GraphLayerItem("up8-1", new Upsampling2D.Builder(2).build(), new String[]{"conv7-2"}));
        graphItemList.add(new GraphLayerItem("up8-2", new ConvolutionLayer.Builder(2,2)
                .stride(1,1)
                .nOut(128)
                .convolutionMode(ConvolutionMode.Same).cudnnAlgoMode(aLgoMode)
                .activation(Activation.RELU).build(), new String[]{"up8-1"}));
        graphItemList.add(new GraphLayerItem("merge8", new MergeVertex(),new String[]{ "conv2-2", "up8-2"}));
        graphItemList.add(new GraphLayerItem("conv8-1", new ConvolutionLayer.Builder(3,3)
                .stride(1,1)
                .nOut(128)
                .convolutionMode(ConvolutionMode.Same).cudnnAlgoMode(aLgoMode)
                .activation(Activation.RELU).build(), new String[]{"merge8"}));
        graphItemList.add(new GraphLayerItem("conv8-2", new ConvolutionLayer.Builder(3,3)
                .stride(1,1)
                .nOut(128)
                .convolutionMode(ConvolutionMode.Same).cudnnAlgoMode(aLgoMode)
                .activation(Activation.RELU).build(), new String[]{"conv8-1"}));

        // up9
        graphItemList.add(new GraphLayerItem("up9-1", new Upsampling2D.Builder(2).build(), new String[]{"conv8-2"}));
        graphItemList.add(new GraphLayerItem("up9-2", new ConvolutionLayer.Builder(2,2)
                .stride(1,1)
                .nOut(64)
                .convolutionMode(ConvolutionMode.Same).cudnnAlgoMode(aLgoMode)
                .activation(Activation.RELU).build(), new String[]{"up9-1"}));
        graphItemList.add(new GraphLayerItem("merge9", new MergeVertex(), new String[]{"conv1-2", "up9-2"}));
        graphItemList.add(new GraphLayerItem("conv9-1", new ConvolutionLayer.Builder(3,3)
                .stride(1,1)
                .nOut(64)
                .convolutionMode(ConvolutionMode.Same)
                .cudnnAlgoMode(aLgoMode)
                .activation(Activation.RELU).build(), new String[]{"merge9"}));
        graphItemList.add(new GraphLayerItem("conv9-2", new ConvolutionLayer.Builder(3,3)
                .stride(1,1)
                .nOut(64)
                .convolutionMode(ConvolutionMode.Same)
                .cudnnAlgoMode(aLgoMode)
                .activation(Activation.RELU).build(), new String[]{"conv9-1"}));
        graphItemList.add(new GraphLayerItem("conv9-3", new ConvolutionLayer.Builder(3,3)
                .stride(1,1)
                .nOut(2)
                .convolutionMode(ConvolutionMode.Same)
                .cudnnAlgoMode(aLgoMode)
                .activation(Activation.RELU).build(), new String[]{"conv9-2"}));

        graphItemList.add(new GraphLayerItem("conv10", new ConvolutionLayer.Builder(1,1)
                .stride(1,1).nOut(3)
                .convolutionMode(ConvolutionMode.Same)
                .cudnnAlgoMode(aLgoMode)
                .activation(Activation.IDENTITY).build(), new String[]{"conv9-3"}));

        return graphItemList;

    }

}
