package org.freeware.dl4j.modelx.train.gan;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.freeware.dl4j.modelx.dataset.cycleGan.CycleGanDataSetIterator;
import org.freeware.dl4j.modelx.dataset.inPainting.InPaintingDataSetIterator;
import org.freeware.dl4j.modelx.model.gan.CycleGan;
import org.freeware.dl4j.modelx.model.gan.InPaintingGan;
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
import org.nd4j.linalg.learning.config.Adam;

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
public class CycleGanTrainer extends AbsGanTrainer{


    private  static Random random=new Random(12345);

    private  static  DataNormalization dataNormalization = new ImagePreProcessingScaler(-1,1);

    private static String outputDir="output_IN_PAINTIN_GAN";

    public static void main(String[] args) {

        //因为目前DL4J还没有实现InstanceNormalization，所以我们只能用batchSize为1时模拟InstanceNormalization
        int batchSize=1;

        int imageHeight =128;

        int imageWidth =128;

        int imageChannel =3;



        String dataPath="dataset/apple2orange";

        CycleGan cycleGan= CycleGan.builder()
                .imageChannel(imageChannel)
                .imageHeight(imageHeight)
                .imageWidth(imageWidth)
                .generatorUpdater(Adam.builder()
                        .learningRate(0.0002)
                        .beta1(0.5).build())
                .discriminatorUpdater(Adam.builder()
                        .learningRate(0.0002)
                        .beta1(0.5).build())
                .build();

        ComputationGraph generatorA2B=cycleGan.initGenerator();

        log.info(generatorA2B.summary());

        ComputationGraph generatorB2A=cycleGan.initGenerator();

        ComputationGraph discriminatorA=cycleGan.initDiscriminator();

        ComputationGraph discriminatorB=cycleGan.initDiscriminator();

        ComputationGraph ganA2B=cycleGan.init();

        ComputationGraph ganB2A=cycleGan.init();

        setListeners(discriminatorA,ganA2B);

        cycleGan.copyParamsFromGanToGeneratorAndDiscriminator(generatorA2B,discriminatorA,ganA2B);

        cycleGan.copyParamsFromGanToGeneratorAndDiscriminator(generatorB2A,discriminatorB,ganB2A);

        MultiDataSetIterator trainData = new CycleGanDataSetIterator(dataPath,batchSize,imageHeight,imageWidth,imageChannel);

        int iterationCounter = 0;

      while (true) {

            trainData.reset();

            while (trainData.hasNext()) {

                iterationCounter++;

                MultiDataSet dataSet= trainData.next();

                INDArray featureA = dataSet.getFeatures()[0];

                dataNormalization.transform(featureA);

                INDArray featureB = dataSet.getLabels()[0];

                dataNormalization.transform(featureB);

                int realBatchSize=(int)featureB.size(0);

                long[] discriminatorOutputShape=new long[]{realBatchSize,1,imageHeight/(2*2*2*2),imageHeight/(2*2*2*2)};

                trainDiscriminator(generatorA2B, discriminatorB, featureA, featureB,discriminatorOutputShape);

                trainDiscriminator(generatorB2A, discriminatorA , featureB,featureA,discriminatorOutputShape);

                cycleGan.copyParamsFromDiscriminatorToGanDiscriminator(discriminatorB, ganA2B);

                cycleGan.copyParamsFromDiscriminatorToGanDiscriminator(discriminatorA, ganB2A);

                trainGan( ganA2B, featureA,discriminatorOutputShape);

                trainGan( ganB2A, featureB,discriminatorOutputShape);

                cycleGan.copyParamsFromGanToGenerator(generatorA2B,ganA2B);

                cycleGan.copyParamsFromGanToGenerator(generatorB2A,ganB2A);

                //visualize(generatorA2B, featureA , iterationCounter);

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
