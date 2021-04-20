package org.freeware.dl4j.modelx.train.gan;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.optimize.listeners.PerformanceListener;
import org.freeware.dl4j.modelx.model.gan.CDCGan;
import org.freeware.dl4j.modelx.utils.DataSetUtils;
import org.freeware.dl4j.modelx.utils.RandomUtils;
import org.freeware.dl4j.modelx.utils.VisualisationUtils;
import org.jetbrains.annotations.NotNull;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.MultiDataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.factory.Nd4j;


import java.io.File;
import java.io.IOException;
import java.util.Random;


/**
 * @author wenfengxu
 * 条件深度卷积生成对抗网络训练
 * 1、把对抗网络的参数复制到生成器和判别器。
 * 2、用真实数据和生成器生成的假数据合并起来训练判别器。
 * 3、把判别器参数复制给对抗网络中的判别器，并冻结对抗网络中的判别器的参数使期不能学习。
 * 4、训练对抗网络，更新对抗网络中的生成器参数，然后把对抗网络中的生成器参数复制给生成器。
 */
@Slf4j
public class CDCGanTrainer {


    private  static Random random=new Random(12345);

    private  static  DataNormalization dataNormalization = new ImagePreProcessingScaler(-1,1);

    public static void main(String[] args) {

        int batchSize=32;

        int numClasses =10;

        int imageHeight =28;

        int imageWidth =28;

        int imageChannel =1;

        String dataPath="/Users/wenfengxu/Downloads/data/mnist_png/training";

        CDCGan cdcgan= CDCGan.builder()
                .numClasses(numClasses)
                .imageChannel(imageChannel)
                .imageHeight(imageHeight)
                .imageWidth(imageWidth)
                .build();

        ComputationGraph generator=cdcgan.initGenerator();

        ComputationGraph discriminator=cdcgan.initDiscriminator();

        ComputationGraph gan=cdcgan.init();

        generator.setListeners(new PerformanceListener(10, true));

        discriminator.setListeners(new PerformanceListener(10, true));

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

                int realBatchSize=Integer.parseInt(String.valueOf(realLabel.size(0)));

                trainDiscriminator(generator, discriminator, realFeature, realLabel);

                cdcgan.copyParamsFromDiscriminatorToGanDiscriminator(discriminator, gan);

                trainGan( gan, realBatchSize);

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

        INDArray[] samples=null;

        if (iterationCounter % 10== 0) {

            samples=getSamples(generator);

            VisualisationUtils.mnistVisualizeForConvolution2D(samples);
        }
        if (iterationCounter % 100== 0) {

            String savePath="output_CDCGAN".concat(File.separator).concat(String.valueOf(iterationCounter));

            VisualisationUtils.saveAsImageForConvolution2D(samples,savePath);
        }
    }


    /**
     * 采样生成器
     * 9个输出
     * @param generator
     * @return
     */
    private static INDArray[] getSamples(ComputationGraph generator) {

        int batchSize=1;

        INDArray[] testSamples = new INDArray[9];

        for(int k=0;k<9;k++){
            //创建batchSize行，100列的随机数浅层空间
            INDArray testLatentDim = Nd4j.rand(new int[]{batchSize,  100});
            //随机标签
            INDArray embeddingLabel= RandomUtils.getRandomEmbeddingLabel(batchSize,0,9,random);
            //输出图片
            INDArray testFakeImaged=generator.output(testLatentDim,embeddingLabel)[0];
            //把图片数据恢复到0-255
            dataNormalization.revertFeatures(testFakeImaged);

            testSamples[k]=testFakeImaged;
        }
        return testSamples;
    }

    /**
     * 训练判别器
     * @param generator
     * @param discriminator
     * @param realFeature
     * @param label
     *
     */
    private static void trainDiscriminator(ComputationGraph generator, ComputationGraph discriminator, INDArray realFeature, INDArray label) {

        int batchSize=Integer.parseInt(String.valueOf(label.size(0)));
        //创建batchSize，100列的随机数浅层空间
        INDArray latentDim = Nd4j.rand(new int[]{batchSize, 100});
        //把标签转为EmbeddingLayer的输入格式
        INDArray embeddingLabel = toEmbeddingFormat(label);
        //用生成器生成假图片，这里的输入标签是使用随机的小批量中获取的，当然也可以自己随机生成
        INDArray fakeImaged=generator.output(latentDim,embeddingLabel)[0];

        realFeature = toImageFormat(realFeature);
        //把生真实的图片和假的图按小批量的维度连接起来
        INDArray fakeAndRealImageFeature= concatOnFirstDimension(realFeature, fakeImaged);
        //把生真实的标签和假的标签按小批量的维度连接起来
        INDArray fakeAndRealLabelFeature= concatOnFirstDimension(embeddingLabel, embeddingLabel);
        //判别器输入特征
        INDArray[] discriminatorFeatures=new INDArray[] {fakeAndRealImageFeature,fakeAndRealLabelFeature};
        //判别器标签 将真假标签按N维度连接后放到标签数组中,注意标签0表示假，1表示真
        INDArray[] discriminatorLabels=new INDArray[] {concatOnFirstDimension(Nd4j.ones(batchSize, 1), Nd4j.zeros(batchSize, 1))};
        //构建多数据集（多个特征，多个标签）
        MultiDataSet discriminatorInputMultiDataSet=new MultiDataSet(discriminatorFeatures,discriminatorLabels);
        //训练判别器
        discriminator.fit(discriminatorInputMultiDataSet);
    }

    /**
     * 接第一个维度拼接
     * @param arrayA
     * @param arrayB
     * @return
     */
    private static INDArray concatOnFirstDimension(INDArray arrayA, INDArray arrayB) {
        return Nd4j.concat(0, arrayA, arrayB);
    }


    /**
     * 对抗训练
     * 此时判别器的学习率为0
     * 所以只会训练生成器
     * @param gan
     * @param batchSize
     *
     */
    private static void trainGan(ComputationGraph gan,  int batchSize) {
        //噪音数据
        INDArray noiseLatentDim = Nd4j.rand(new int[]{batchSize, 100});
        //随机长成EmbeddingLayer输入格式的标签
        INDArray embeddingLabel = RandomUtils.getRandomEmbeddingLabel(batchSize,0,9,random);
        //噪音特征
        INDArray[] noiseLatentFeature = new INDArray[]{noiseLatentDim, embeddingLabel};
        //这里故意把噪音的标签设为真，
        INDArray[] noiseLatentLabel = new INDArray[]{Nd4j.ones(batchSize, 1)};
        //对抗网络输入多数据集
        MultiDataSet ganInputMultiDataSet = new MultiDataSet(noiseLatentFeature, noiseLatentLabel);

        gan.fit(ganInputMultiDataSet);

    }



    /**
     * 转为图片格式
     * @param feature
     * @return
     */
    private static INDArray toImageFormat(INDArray feature) {
        return feature.reshape(feature.size(0),1,28,28);
    }

    /**
     * 转换为EmbeddingLayer的输入格式
     * [batchSize,labelIndex]
     * @param label
     * @return
     */
    private static INDArray toEmbeddingFormat(INDArray label) {

        INDArray maxIdx=label.argMax(1);

        INDArray embeddingLabel= Nd4j.expandDims(maxIdx,1);

        return embeddingLabel;
    }
}
