package org.freeware.dl4j.modelx.train.gan;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.optimize.listeners.PerformanceListener;
import org.freeware.dl4j.modelx.model.gan.CGan;
import org.freeware.dl4j.modelx.utils.VisualisationUtils;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.MultiDataSet;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.factory.Nd4j;

import java.io.IOException;


@Slf4j
public class CGanTrainer {




    public static void main(String[] args) {

        CGan cgan= CGan.builder().build();

        ComputationGraph generator=cgan.initGenerator();

        ComputationGraph discriminator=cgan.initDiscriminator();

        ComputationGraph gan=cgan.init();

        generator.setListeners(new PerformanceListener(10, true));

        discriminator.setListeners(new PerformanceListener(10, true));

        log.info(generator.summary());

        log.info(discriminator.summary());

        log.info(gan.summary());

        Nd4j.getMemoryManager().setAutoGcWindow(15 * 1000);

        MnistDataSetIterator trainData = null;
        try {
            trainData = new MnistDataSetIterator(128, true, 42);
        } catch (IOException e) {
            e.printStackTrace();
        }

        DataNormalization dataNormalization = new ImagePreProcessingScaler(-1, 1);

        trainData.setPreProcessor(dataNormalization);

        dataNormalization.fit(trainData);

        while (true) {

            trainData.reset();

            int iterationCounter = 0;

            cgan.copyParamsFromGanToGeneratorAndDiscriminator(generator,discriminator,gan);

            while (trainData.hasNext()) {

                iterationCounter++;

                DataSet dataSet=trainData.next();

                INDArray realFeature = dataSet.getFeatures();

                INDArray realLabel = dataSet.getLabels();

                int batchSize = (int) realFeature.size(0);

                trainDiscriminator(generator, discriminator, realFeature, realLabel, batchSize);

                trainGan(cgan, generator, discriminator, gan, realLabel, batchSize);


                if (iterationCounter % 10 == 1) {

                    visualize(generator, realLabel, batchSize);

                }

            }

        }



    }

    private static void visualize(ComputationGraph generator, INDArray realLabel, int batchSize) {

        INDArray[] samples = new INDArray[9];

        for(int k=0;k<9;k++){
            //创建batchSize行，100列的随机数浅层空间
            INDArray testLatentDim = Nd4j.rand(new int[]{batchSize,  100});

            INDArray testFakeImaged=generator.output(testLatentDim,realLabel)[0];

            samples[k]=testFakeImaged;
        }

        VisualisationUtils.mnistVisualize(samples);
    }

    private static void trainDiscriminator(ComputationGraph generator, ComputationGraph discriminator, INDArray realFeature, INDArray realLabel, int batchSize) {
        //创建batchSize行，100列的随机数浅层空间
        INDArray latentDim = Nd4j.rand(new int[]{batchSize,  100});
        //用生成器生成假图片，这里的输入标签是使用随机的小批量中获取的，当然也可以自己随机生成
        INDArray fakeImaged=generator.output(latentDim,realLabel)[0];
        //把生真实的图片和假的图按小批量的维度连接起来
        INDArray fakeAndRealImageFeature=Nd4j.concat(0,realFeature,fakeImaged);
        //把生真实的标签和假的标签按小批量的维度连接起来
        INDArray fakeAndRealLabelFeature=Nd4j.concat(0,realLabel,realLabel);
        //判别器输入特征
        INDArray[] discriminatorFeatures=new INDArray[] {fakeAndRealImageFeature,fakeAndRealLabelFeature};
        //判别器标签 将真假标签按N维度连接后放到标签数组中,注意标签0表示假，1表示真
        INDArray[] discriminatorLabels=new INDArray[] {Nd4j.concat(0,Nd4j.ones(batchSize, 1),Nd4j.zeros(batchSize, 1))};
        //构建多数据集（多个特征，多个标签）
        MultiDataSet discriminatorInputMultiDataSet=new MultiDataSet(discriminatorFeatures,discriminatorLabels);
        //训练判别器
        discriminator.fit(discriminatorInputMultiDataSet);
    }

    private static void trainGan(CGan cgan, ComputationGraph generator, ComputationGraph discriminator, ComputationGraph gan, INDArray realLabel, int batchSize) {

        cgan.copyParamsFromDiscriminatorToGanDiscriminator(discriminator, gan);

        INDArray noiseLatentDim = Nd4j.rand(new int[]{batchSize, 100});
        //噪音特征
        INDArray[] noiseLatentFeatureArray = new INDArray[]{noiseLatentDim, realLabel};
        //这里故意把噪音的标签设为真，
        INDArray[] noiseLatentLabelArray = new INDArray[]{Nd4j.ones(batchSize, 1)};

        MultiDataSet ganInputMultiDataSet = new MultiDataSet(noiseLatentFeatureArray, noiseLatentLabelArray);

        gan.fit(ganInputMultiDataSet);

        cgan.copyParamsFromGanToGenerator(generator,gan);
    }

}
