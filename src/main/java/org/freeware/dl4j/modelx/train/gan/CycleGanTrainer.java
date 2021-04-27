package org.freeware.dl4j.modelx.train.gan;

import lombok.extern.slf4j.Slf4j;
import org.apache.commons.lang3.ArrayUtils;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.freeware.dl4j.modelx.dataset.cycleGan.CycleGanDataSetIterator;
import org.freeware.dl4j.modelx.model.gan.CycleGan;
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
 * CycleGan网络训练
 * CycleGan不需要成对的训练数据即可实现风格转移
 *
 *
 *
 */
@Slf4j
public class CycleGanTrainer extends AbsGanTrainer{


    private  static Random random=new Random(12345);

    private  static  DataNormalization dataNormalization = new ImagePreProcessingScaler(-1,1);

    private static String outputDir="output_Cycle_GAN";

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

        ComputationGraph generatorB2A=cycleGan.initGenerator();

        ComputationGraph discriminatorA=cycleGan.initDiscriminator();

        ComputationGraph discriminatorB=cycleGan.initDiscriminator();

        ComputationGraph ganA2B=cycleGan.init();

        ComputationGraph ganB2A=cycleGan.init();

        ComputationGraph reconstructA2B2A=cycleGan.initReconstructNetwork();

        ComputationGraph reconstructB2A2B=cycleGan.initReconstructNetwork();

        ComputationGraph identityMappingNetworkA2B=cycleGan.initIdentityMappingNetwork();

        ComputationGraph identityMappingNetworkB2A=cycleGan.initIdentityMappingNetwork();

        setListeners(discriminatorA,discriminatorB,ganA2B,ganB2A,reconstructA2B2A,reconstructB2A2B,identityMappingNetworkA2B,identityMappingNetworkB2A);

        cycleGan.copyParamsFromGanToGeneratorAndDiscriminator(generatorA2B,discriminatorB,ganA2B);

        cycleGan.copyParamsFromGanToGeneratorAndDiscriminator(generatorB2A,discriminatorA,ganB2A);

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

                copyParamsToReconstructNetwork(reconstructA2B2A,ganA2B,ganB2A);

                trainReconstructNetwork(reconstructA2B2A,featureA);

                copyParamsFromA2B2AtoB2A2B(reconstructA2B2A,reconstructB2A2B);

                trainReconstructNetwork(reconstructB2A2B,featureB);

                copyParamsFromB2A2BtoIdentityMappingNetwork(reconstructB2A2B,identityMappingNetworkA2B,identityMappingNetworkB2A);

                trainIdentityMappingNetwork(identityMappingNetworkB2A,featureA);

                trainIdentityMappingNetwork(identityMappingNetworkA2B,featureB);

                copyParamsFromIdentityMappingNetworkToGenerator(identityMappingNetworkA2B,generatorA2B);

                copyParamsFromIdentityMappingNetworkToGenerator(identityMappingNetworkB2A,generatorB2A);

                visualize(generatorA2B, generatorB2A,featureA , featureB,iterationCounter);



            }

        }


    }


    /**
     * 把对抗网络ganA2B、ganB2A中的
     * 生成器参数复制给
     * 重建网络A2B2A中的生成器A2B和B2A
     * @param reconstructA2B2A
     * @param ganA2B
     * @param ganB2A
     */
    private static void copyParamsToReconstructNetwork(ComputationGraph reconstructA2B2A,ComputationGraph ganA2B,ComputationGraph ganB2A){
        int halfGenLayerLen = reconstructA2B2A.getLayers().length/2;
        for (int i = 0; i < halfGenLayerLen; i++) {
            reconstructA2B2A.getLayer(i).setParams(ganA2B.getLayer(i).params());
            reconstructA2B2A.getLayer(i+halfGenLayerLen).setParams(ganB2A.getLayer(i).params());
        }
    }

    /**
     * 把重建网络A2B2A中的生成器A2B和B2A
     * 参数复制给把重建网络B2A2B中的生成器B2A和A2B
     * @param reconstructA2B2A
     * @param reconstructB2A2B
     */
    private static void copyParamsFromA2B2AtoB2A2B(ComputationGraph reconstructA2B2A,ComputationGraph reconstructB2A2B){
        int halfGenLayerLen = reconstructA2B2A.getLayers().length/2;
        for (int i = 0; i < halfGenLayerLen; i++) {
            reconstructB2A2B.getLayer(i).setParams(reconstructA2B2A.getLayer(halfGenLayerLen+i).params());
            reconstructB2A2B.getLayer(i+halfGenLayerLen).setParams(reconstructA2B2A.getLayer(i).params());
        }
    }

    /**
     * 把重建网络B2A2B中的生成器B2A和A2B参数
     * 复制给恒等网络A2B和B2A
     * @param reconstructB2A2B
     * @param identityMappingNetworkA2B
     * @param identityMappingNetworkB2A
     */
    private static void copyParamsFromB2A2BtoIdentityMappingNetwork(ComputationGraph reconstructB2A2B, ComputationGraph identityMappingNetworkA2B, ComputationGraph identityMappingNetworkB2A){
        int halfGenLayerLen = reconstructB2A2B.getLayers().length/2;
        for (int i = 0; i < halfGenLayerLen; i++) {
            identityMappingNetworkB2A.getLayer(i).setParams(reconstructB2A2B.getLayer(i).params());
            identityMappingNetworkA2B.getLayer(i).setParams(reconstructB2A2B.getLayer(halfGenLayerLen+i).params());
        }
    }


    /**
     * 把恒等网络中的参数复制给生成器
     * @param identityMappingNetwork
     * @param generator
     */
    private static void copyParamsFromIdentityMappingNetworkToGenerator(ComputationGraph identityMappingNetwork, ComputationGraph generator){
        int len = generator.getLayers().length;
        for (int i = 0; i < len; i++) {
            generator.getLayer(i).setParams(identityMappingNetwork.getLayer(i).params());
        }
    }
    /**
     * 可视化
     * @param generatorA2B
     * @param generatorB2A
     * @param iterationCounter
     */
    private static void visualize(ComputationGraph generatorA2B,ComputationGraph generatorB2A,  INDArray featureA ,INDArray featureB ,int iterationCounter) {

        Sample[] samples=null;

        if (iterationCounter % 10== 0) {

            Sample[] samplesA=getSamples(generatorA2B,featureA);

            Sample[] samplesB=getSamples(generatorB2A,featureB);

            samples= ArrayUtils.addAll(samplesA,samplesB);

            VisualisationUtils.visualizeForConvolution2D(samples,"CycleGan");
        }
        if (iterationCounter % 1000== 0) {

            String savePath=outputDir.concat(File.separator).concat(String.valueOf(iterationCounter));

            VisualisationUtils.saveAsImageForConvolution2D(samples,savePath);
        }
    }


    /**
     * 采样生成器
     * @param generator
     * @return
     */
    private static Sample[] getSamples(ComputationGraph generator, INDArray feature) {

        int batchSize=(int)feature.size(0);
        //输入+输出
        int sampleLen=batchSize*2;

        Sample[] samples = new Sample[sampleLen];

        for(int k=0;k<batchSize;k++){
            //创建batchSize行，100列的随机数浅层空间
            INDArray latentDim =feature.get(new INDArrayIndex[]{NDArrayIndex.point(k),NDArrayIndex.all(),NDArrayIndex.all(),NDArrayIndex.all()});

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
