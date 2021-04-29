package org.freeware.dl4j.modelx.train.gan;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.freeware.dl4j.modelx.dataset.inPainting.InPaintingDataSetIterator;
import org.freeware.dl4j.modelx.model.gan.InPaintingGan;
import org.freeware.dl4j.modelx.model.gan.InPaintingGanV2;
import org.freeware.dl4j.modelx.utils.Sample;
import org.freeware.dl4j.modelx.utils.VisualisationUtils;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.learning.config.RmsProp;
import org.nd4j.linalg.learning.config.Sgd;

import java.io.File;
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
public class InPaintingGanV2Trainer extends AbsGanTrainer{


    private  static Random random=new Random(12345);

    private  static  DataNormalization dataNormalization = new ImagePreProcessingScaler(-1,1);

    private static String outputDir="output_IN_PAINTIN_GAN_V2";

    public static void main(String[] args) {

        //因为目前DL4J还没有实现InstanceNormalization，所以我们只能用batchSize为1时模拟InstanceNormalization
        int batchSize=1;

        int imageHeight =512;

        int imageWidth =512;

        int imageChannel =3;

        String dataPath="dataset/inpainting";

        InPaintingGanV2 inPaintingGan= InPaintingGanV2.builder()
                .imageChannel(imageChannel)
                .imageHeight(imageHeight)
                .imageWidth(imageWidth)
                .generatorUpdater(new Sgd(4E-4))
                .discriminatorUpdater( new RmsProp(0.1, 0.96, 0.001))
                .build();

        ComputationGraph generator=inPaintingGan.initGenerator();

        ComputationGraph discriminator=inPaintingGan.initDiscriminator();

        ComputationGraph gan=inPaintingGan.init();

        setListeners(discriminator,gan);
        log.info(discriminator.summary());
        log.info(gan.summary());

        inPaintingGan.copyParamsFromGanToGeneratorAndDiscriminator(generator,discriminator,gan);

        Nd4j.getMemoryManager().setAutoGcWindow(15 * 1000);

        MultiDataSetIterator trainData = new InPaintingDataSetIterator(dataPath,batchSize,imageHeight,imageWidth,imageChannel);

        int iterationCounter = 0;

      while (true) {

            trainData.reset();

            while (trainData.hasNext()) {

                iterationCounter++;

                MultiDataSet dataSet= trainData.next();

                INDArray realFeature = dataSet.getFeatures()[0];

                dataNormalization.transform(realFeature);

                INDArray realLabel = dataSet.getLabels()[0];

                dataNormalization.transform(realLabel);

                int realBatchSize=(int)realLabel.size(0);

                trainDiscriminator(generator, discriminator, realFeature, realLabel,realBatchSize);

                inPaintingGan.copyParamsFromDiscriminatorToGanDiscriminator(discriminator, gan);

                trainGan( gan, realBatchSize,realFeature);

                inPaintingGan.copyParamsFromGanToGenerator(generator,gan);

                visualize(generator, realFeature , iterationCounter);

            }

        }

    }

    /**
     * 可视化
     * @param generator
     * @param iterationCounter
     */
    private static void visualize(ComputationGraph generator, INDArray realFeature ,int iterationCounter) {

        Sample[] samples=null;

        if (iterationCounter % 10== 0) {

            samples=getSamples(generator,realFeature);

            VisualisationUtils.visualizeForConvolution2D(samples,"InPaintingGan");
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
    private static Sample[] getSamples(ComputationGraph generator, INDArray realFeature) {

        int batchSize=(int)realFeature.size(0);
        //输入+输出
        int sampleLen=batchSize*2;

        Sample[] samples = new Sample[sampleLen];

        for(int k=0;k<batchSize;k++){
            //创建batchSize行，100列的随机数浅层空间
            INDArray latentDim =realFeature.get(new INDArrayIndex[]{NDArrayIndex.point(k),NDArrayIndex.all(),NDArrayIndex.all(),NDArrayIndex.all()});

            latentDim=Nd4j.expandDims(latentDim,0);
            //输出图片
            INDArray fakeImage=generator.output(latentDim)[0];
            //把图片数据恢复到0-255
            dataNormalization.revertFeatures(latentDim);

            dataNormalization.revertFeatures(fakeImage);

            Sample sampleInput=new Sample(latentDim,"input");

            Sample sampleOutput=new Sample(fakeImage,"output");

            samples[k]=sampleInput;

            samples[k+1]=sampleOutput;
        }
        return samples;
    }







}
