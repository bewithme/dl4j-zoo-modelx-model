package org.freeware.dl4j.modelx.train.gan;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.objdetect.Yolo2OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.deeplearning4j.util.ModelSerializer;
import org.freeware.dl4j.modelx.dataset.srgan.SrGanDataSetIterator;
import org.freeware.dl4j.modelx.model.gan.SRGan;
import org.freeware.dl4j.modelx.utils.*;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.io.File;
import java.io.IOException;
import java.util.Random;


/**
 * @author wenfengxu
 * 条件深度卷积生成对抗网络训练
 * 1、把对抗网络的参数复制到生成器和判别器。
 * 2、用真实数据和生成器生成的假数据训练判别器。
 * 3、把判别器参数复制给对抗网络中的判别器，并冻结对抗网络中的判别器的参数使它不能学习。
 * 4、训练对抗网络，更新对抗网络中的生成器参数，然后把对抗网络中的生成器参数复制给生成器。
 */
@Slf4j
public class SRGanTrainer extends AbsGanTrainer{


    private  static Random random=new Random(12345);

    private  static  DataNormalization dataNormalization = new ImagePreProcessingScaler(-1,1);

    private static String outputDir="output_SRGAN";

    public static void main(String[] args) {

        int batchSize=1;

        int imageHeight =64;

        int imageWidth =64;

        int imageChannel =3;

        int imageHeightHr =256;

        int imageWidthHr =256;

        int imageChannelHr =3;

        String dataPath="/Volumes/feng/ai/dataset/VGG-Face2/data/test";

        String encoderFilePath="models/cae/caeEncoder.zip";

        SRGan srgan= SRGan.builder()
                .imageChannel(imageChannel)
                .imageHeight(imageHeight)
                .imageWidth(imageWidth)
                .build();

        ComputationGraph generator=srgan.initGenerator();

        ComputationGraph discriminator=srgan.initDiscriminator();

        ComputationGraph gan=srgan.init();

        ComputationGraph encoder=null;

        try {
             encoder= ModelSerializer.restoreComputationGraph(encoderFilePath,Boolean.FALSE);
        } catch (IOException e) {
            log.info("",e);
        }

        setListeners(generator,discriminator,gan);

        log.info(generator.summary());
        log.info(gan.summary());

        srgan.copyParamsWhenFromIsPartOfToByName(encoder,gan);

        MultiDataSetIterator srGanMultiDataSetIterator=new SrGanDataSetIterator(dataPath,batchSize,imageHeight,imageHeight,imageChannel,imageHeightHr,imageWidthHr,imageChannelHr);

        int iterationCounter = 0;

        while (true) {

            srGanMultiDataSetIterator.reset();

            while (srGanMultiDataSetIterator.hasNext()) {

                iterationCounter++;

                MultiDataSet dataSet= srGanMultiDataSetIterator.next();

                INDArray features = dataSet.getFeatures()[0];

                dataNormalization.transform(features);
                //256x256
                INDArray label = dataSet.getLabels()[0];

                dataNormalization.transform(label);

                int realBatchSize=(int)features.size(0);

                trainDiscriminator(generator,discriminator,features,label,realBatchSize);

                srgan.copyParamsFromDiscriminatorToGanDiscriminatorByName(discriminator, gan);

                INDArray mapFeature=encoder.output(false,label)[0];

                trainGan(gan,realBatchSize,features,mapFeature);

                srgan.copyParamsFromGanToGeneratorByName(generator,gan);

                visualize(generator, iterationCounter,features);

            }

        }

    }



    public static void trainGan(ComputationGraph gan,int batchSize,INDArray generatorInput,INDArray mapFeature ) {
        //噪音特征
        INDArray[] noiseLatentFeature = new INDArray[]{generatorInput};
        //这里故意把噪音的标签设为真，
        INDArray[] noiseLatentLabel = new INDArray[]{Nd4j.ones(batchSize, 1),mapFeature};
        //对抗网络输入多数据集
        org.nd4j.linalg.dataset.MultiDataSet ganInputMultiDataSet = new org.nd4j.linalg.dataset.MultiDataSet(noiseLatentFeature, noiseLatentLabel);

        gan.fit(ganInputMultiDataSet);

    }



    /**
     * 可视化
     * @param generator
     * @param iterationCounter
     */
    private static void visualize(ComputationGraph generator, int iterationCounter,INDArray features) {

        Sample[] samples=null;

        if (iterationCounter % 10== 0) {

            samples=getSamples(generator,features);

            VisualisationUtils.visualizeForConvolution2D(samples,"SRGan");
        }
        if (iterationCounter % 1000== 0) {

            String savePath=outputDir.concat(File.separator).concat(String.valueOf(iterationCounter));

            VisualisationUtils.saveAsImageForConvolution2D(samples,savePath);
        }
    }


    /**
     * 采样生成器
     * 9个输出
     * @param generator
     * @return
     */
    private static Sample[] getSamples(ComputationGraph generator,INDArray features) {

        int batchSize=getBatchSize(features);

        batchSize=clipBatchSize(batchSize,3);
        //输入+输出
        int sampleLen=batchSize*2;

        Sample[] samples = new Sample[sampleLen];

        for(int k=0;k<batchSize;k++){

            INDArray latentDim =features.get(new INDArrayIndex[]{NDArrayIndex.point(k),NDArrayIndex.all(),NDArrayIndex.all(),NDArrayIndex.all()});

            latentDim=Nd4j.expandDims(latentDim,0);
            //输出图片
            INDArray fakeImage=generator.output(latentDim)[0];
            //把图片数据恢复到0-255
            dataNormalization.revertFeatures(latentDim);

            dataNormalization.revertFeatures(fakeImage);

            Sample sampleInput=new Sample(latentDim,"input");

            Sample sampleOutput=new Sample(fakeImage,"output");

            samples[k*2]=sampleInput;

            samples[k*2+1]=sampleOutput;
        }
        return samples;
    }







}
