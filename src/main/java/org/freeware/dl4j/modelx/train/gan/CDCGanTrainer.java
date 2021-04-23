package org.freeware.dl4j.modelx.train.gan;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.freeware.dl4j.modelx.model.gan.CDCGan;
import org.freeware.dl4j.modelx.utils.Sample;
import org.freeware.dl4j.modelx.utils.*;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;


import java.io.File;
import java.io.IOException;
import java.util.Random;


/**
 * @author wenfengxu
 * 条件深度卷积生成对抗网络训练
 * 1、把对抗网络的参数复制到生成器和判别器。
 * 2、用真实数据和生成器生成的假数据训练判别器。
 * 3、把判别器参数复制给对抗网络中的判别器，并冻结对抗网络中的判别器的参数使期不能学习。
 * 4、训练对抗网络，更新对抗网络中的生成器参数，然后把对抗网络中的生成器参数复制给生成器。
 */
@Slf4j
public class CDCGanTrainer extends AbsGanTrainer{


    private  static Random random=new Random(12345);

    private  static  DataNormalization dataNormalization = new ImagePreProcessingScaler(-1,1);

    private static String outputDir="output_CDCGAN";

    public static void main(String[] args) {

        int batchSize=32;

        int numClasses =10;

        int imageHeight =28;

        int imageWidth =28;

        int imageChannel =1;

        int latentDimLen=100;


        String dataPath="/Users/wenfengxu/Downloads/data/mnist_png/training";

        CDCGan cdcgan= CDCGan.builder()
                .numClasses(numClasses)
                .imageChannel(imageChannel)
                .imageHeight(imageHeight)
                .imageWidth(imageWidth)
                .updater(Adam.builder()
                        .learningRate(0.0003)
                        .beta1(0.5).build()
                 )
                .build();

        ComputationGraph generator=cdcgan.initGenerator();

        ComputationGraph discriminator=cdcgan.initDiscriminator();

        ComputationGraph gan=cdcgan.init();

        setListeners(discriminator,gan);

        cdcgan.copyParamsFromGanToGeneratorAndDiscriminator(generator,discriminator,gan);

        log.info(gan.summary());

        Nd4j.getMemoryManager().setAutoGcWindow(15 * 1000);

        DataSetIterator trainData = null;

        try {

            trainData=DataSetUtils.getDataSetIterator(dataPath,batchSize,10,imageHeight,imageWidth,imageChannel);

        } catch (IOException e) {
            e.printStackTrace();
        }

        int iterationCounter = 0;

        while (true) {

            trainData.reset();

            while (trainData.hasNext()) {

                iterationCounter++;

                DataSet dataSet=trainData.next();

                INDArray realFeature = dataSet.getFeatures();

                INDArray realLabel = dataSet.getLabels();

                int realBatchSize=(int)realLabel.size(0);

                trainDiscriminator(generator, discriminator, realFeature, realLabel,realBatchSize,latentDimLen,0,9,random);

                cdcgan.copyParamsFromDiscriminatorToGanDiscriminator(discriminator, gan);

                trainGan( gan, realBatchSize,latentDimLen,0,9,random);

                cdcgan.copyParamsFromGanToGenerator(generator,gan);

                visualize(generator, iterationCounter);

            }

        }

    }

    /**
     * 可视化
     * @param generator
     * @param iterationCounter
     */
    private static void visualize(ComputationGraph generator, int iterationCounter) {

        Sample[] samples=null;

        if (iterationCounter % 10== 0) {

            samples=getSamples(generator);

            VisualisationUtils.mnistVisualizeForConvolution2D(samples,"CDCGan");
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
    private static Sample[] getSamples(ComputationGraph generator) {

        int batchSize=1;

        Sample[] samples = new Sample[9];

        for(int k=0;k<9;k++){
            //创建batchSize行，100列的随机数浅层空间
            INDArray latentDim = Nd4j.rand(new int[]{batchSize,  100});
            //随机标签
            INDArray fakeEmbeddingLabel= RandomUtils.getRandomEmbeddingLabel(batchSize,0,9,random);
            //输出图片
            INDArray fakeImage=generator.output(latentDim,fakeEmbeddingLabel)[0];
            //把图片数据恢复到0-255
            dataNormalization.revertFeatures(fakeImage);

            Sample sample=new Sample(fakeImage,String.valueOf(fakeEmbeddingLabel.toIntVector()[0]));

            samples[k]=sample;
        }
        return samples;
    }







}
