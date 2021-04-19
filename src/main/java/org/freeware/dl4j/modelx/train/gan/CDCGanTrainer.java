package org.freeware.dl4j.modelx.train.gan;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.optimize.listeners.PerformanceListener;
import org.freeware.dl4j.modelx.model.gan.CDCGan;
import org.freeware.dl4j.modelx.utils.INDArrayUtils;
import org.freeware.dl4j.modelx.utils.VisualisationUtils;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.MultiDataSet;
import org.nd4j.linalg.factory.Nd4j;


import java.io.File;
import java.io.IOException;


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

    public static void main(String[] args) {

        int batchSize=128;

        CDCGan cdcgan= CDCGan.builder().build();

        ComputationGraph generator=cdcgan.initGenerator();

        ComputationGraph discriminator=cdcgan.initDiscriminator();

        ComputationGraph gan=cdcgan.init();

        generator.setListeners(new PerformanceListener(10, true));

        discriminator.setListeners(new PerformanceListener(10, true));

        cdcgan.copyParamsFromGanToGeneratorAndDiscriminator(generator,discriminator,gan);

        log.info(generator.summary());

        log.info(discriminator.summary());

        log.info(gan.summary());

        Nd4j.getMemoryManager().setAutoGcWindow(15 * 1000);

        MnistDataSetIterator trainData = null;
        try {
            trainData = new MnistDataSetIterator(batchSize, true, 42);
        } catch (IOException e) {
            e.printStackTrace();
        }


        while (true) {

            trainData.reset();

            int iterationCounter = 0;


            while (trainData.hasNext()) {

                iterationCounter++;

                DataSet dataSet=trainData.next();

                INDArray realFeature = dataSet.getFeatures();

                INDArray realLabel = dataSet.getLabels();

                trainDiscriminator(generator, discriminator, realFeature, realLabel);

                cdcgan.copyParamsFromDiscriminatorToGanDiscriminator(discriminator, gan);

                trainGan( gan, realLabel);

                cdcgan.copyParamsFromGanToGenerator(generator,gan);

                if (iterationCounter % 1000== 1) {

                    visualize(generator, realLabel, iterationCounter);

                }

            }

        }



    }

    /**
     * 测试数据可视化
     * @param generator 生成器
     * @param label 随机标签
     *
     * @param iterationCounter 迭代次数
     */
    private static void visualize(ComputationGraph generator, INDArray label,int iterationCounter) {

        int batchSize=Integer.parseInt(String.valueOf(label.size(0)));

        INDArray[] testSamples = new INDArray[9];

        for(int k=0;k<9;k++){
            //创建batchSize行，100列的随机数浅层空间
            INDArray testLatentDim = Nd4j.rand(new int[]{batchSize,  100});

            INDArray embeddingLabel=toEmbeddingFormat(label);

            INDArray testFakeImaged=generator.output(testLatentDim,embeddingLabel)[0];

            testSamples[k]=testFakeImaged;
        }

        String savePath="output_cdcgan".concat(File.separator).concat(String.valueOf(iterationCounter));

        VisualisationUtils.saveAsImage(testSamples,savePath);

        VisualisationUtils.mnistVisualize(testSamples);
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


        label = INDArrayUtils.getHalfOfFirstDimension(label);

        int batchSize=Integer.parseInt(String.valueOf(label.size(0)));

        //创建batchSize/2 行，100列的随机数浅层空间
        INDArray latentDim = Nd4j.rand(new int[]{batchSize, 100});

        INDArray embeddingLabel = toEmbeddingFormat(label);
        //用生成器生成假图片，这里的输入标签是使用随机的小批量中获取的，当然也可以自己随机生成
        INDArray fakeImaged=generator.output(latentDim,embeddingLabel)[0];

        realFeature = toImageFormat(realFeature);

        realFeature = INDArrayUtils.getHalfOfFirstDimension(realFeature);
        //把生真实的图片和假的图按小批量的维度连接起来
        INDArray fakeAndRealImageFeature=Nd4j.concat(0,realFeature,fakeImaged);
        //把生真实的标签和假的标签按小批量的维度连接起来
        INDArray fakeAndRealLabelFeature=Nd4j.concat(0,embeddingLabel,embeddingLabel);
        //判别器输入特征
        INDArray[] discriminatorFeatures=new INDArray[] {fakeAndRealImageFeature,fakeAndRealLabelFeature};
        //判别器标签 将真假标签按N维度连接后放到标签数组中,注意标签0表示假，1表示真
        INDArray[] discriminatorLabels=new INDArray[] {Nd4j.concat(0,Nd4j.ones(batchSize, 1),Nd4j.zeros(batchSize, 1))};
        //构建多数据集（多个特征，多个标签）
        MultiDataSet discriminatorInputMultiDataSet=new MultiDataSet(discriminatorFeatures,discriminatorLabels);
        //训练判别器
        discriminator.fit(discriminatorInputMultiDataSet);
    }



    private static INDArray toImageFormat(INDArray feature) {
        return feature.reshape(feature.size(0),1,28,28);
    }

    private static INDArray toEmbeddingFormat(INDArray label) {

        INDArray maxIdx=label.argMax(1);

        INDArray embeddingLabel= Nd4j.expandDims(maxIdx,1);

        return embeddingLabel;
    }

    /**
     * 对抗训练
     * @param gan
     * @param label
     *
     */
    private static void trainGan(ComputationGraph gan, INDArray label) {


        int batchSize=Integer.parseInt(String.valueOf(label.size(0)));

        INDArray noiseLatentDim = Nd4j.rand(new int[]{batchSize, 100});

        INDArray maxIdx = toEmbeddingFormat(label);

        //噪音特征
        INDArray[] noiseLatentFeature = new INDArray[]{noiseLatentDim, maxIdx};
        //这里故意把噪音的标签设为真，
        INDArray[] noiseLatentLabel = new INDArray[]{Nd4j.ones(batchSize, 1)};

        MultiDataSet ganInputMultiDataSet = new MultiDataSet(noiseLatentFeature, noiseLatentLabel);

        gan.fit(ganInputMultiDataSet);

    }

}
