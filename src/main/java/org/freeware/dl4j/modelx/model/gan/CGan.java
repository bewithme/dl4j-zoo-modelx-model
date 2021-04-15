package org.freeware.dl4j.modelx.model.gan;

import lombok.extern.slf4j.Slf4j;
import org.apache.commons.lang3.ArrayUtils;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.PerformanceListener;
import org.freeware.dl4j.modelx.utils.RandomUtils;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.activations.impl.ActivationLReLU;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.learning.config.IUpdater;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.util.Arrays;
import java.util.Random;

/**
 * 条件生成对抗网络示例
 * 原理：潜层空间+标签（one hot或embedding） 作为生成器的输入
 *
 *
 */

@Slf4j
public class CGan {
    private static final double LEARNING_RATE = 0.0002;
    //private static final double L2 = 0.005;
    private static final double GRADIENT_THRESHOLD = 100.0;
    private static final IUpdater UPDATER = Adam.builder().learningRate(LEARNING_RATE).beta1(0.5).build();
    private static final IUpdater UPDATER_ZERO = Sgd.builder().learningRate(0.0).build();

    private  static int DISCRIMINATOR_INPUT_SIZE =784;

    private  static int GENERATOR_INPUT_SIZE =100+10;
    private static JFrame frame;
    private static JPanel panel;
    private static Random random =new Random(123456);
    public static void main(String... args) throws Exception {
       
    	
    	Nd4j.getMemoryManager().setAutoGcWindow(15 * 1000);

        MnistDataSetIterator trainData = new MnistDataSetIterator(128, true, 42);

        DataNormalization dataNormalization = new ImagePreProcessingScaler(-1, 1);

        trainData.setPreProcessor(dataNormalization);

		MultiLayerNetwork generator = new MultiLayerNetwork(buildGenerator());

        MultiLayerNetwork discriminator = new MultiLayerNetwork(buildDiscriminator());

        MultiLayerNetwork gan = new MultiLayerNetwork(buildGan());

        generator.init();

        discriminator.init();

        gan.init();
        //从GAN把参数复制到生成器和判别器
        copyParamsToGeneratorAndDiscriminatorFromGan(generator, discriminator, gan);

        generator.setListeners(new PerformanceListener(10, true));

        discriminator.setListeners(new PerformanceListener(10, true));

        gan.setListeners(new PerformanceListener(10, true));

        while (true) {

            trainData.reset();

        	int j = 0;
           
            while (trainData.hasNext()) {
                j++;

                DataSet dataSet=trainData.next();

                INDArray realFeature = dataSet.getFeatures();

                INDArray realLabel = dataSet.getLabels();

                int batchSize = (int) realFeature.size(0);
                //训练判别器
                INDArray combinedLatentDim = trainDiscriminator(generator, discriminator, gan, realFeature, realLabel, batchSize);

                if (j % 10 == 1) {
                    //对抗训练
                    trainGan(generator, discriminator, gan, realLabel, batchSize);

                    visualize(generator, gan, batchSize, combinedLatentDim);
                }
            }

        }
    }

    private static void visualize(MultiLayerNetwork generator, MultiLayerNetwork gan, int batchSize, INDArray combinedLatentDim) {

        INDArray[] samples = new INDArray[9];

        DataSet fakeSetTest = new DataSet(combinedLatentDim, Nd4j.ones(batchSize, 1));

        //取出9条随机潜层空间数据
        for (int k = 0; k < 9; k++) {

            INDArray input = fakeSetTest.get(k).getFeatures();
            //也可以使用 samples[k] = gen.output(input, false);
            samples[k] = gan.activateSelectedLayers(0, generator.getLayers().length - 1, input);

        }
        visualize(samples);
    }

    private static INDArray trainDiscriminator(MultiLayerNetwork generator, MultiLayerNetwork discriminator, MultiLayerNetwork gan, INDArray realFeature, INDArray realLabel, int batchSize) {
        //创建batchSize行，100列的随机数浅层空间
        INDArray latentDim = Nd4j.rand(new int[]{batchSize,  100});
        //将100列的随机数浅层空间和随机的标签结合起来作为生成器的输入
        INDArray combinedLatentDim=Nd4j.concat(-1,latentDim,realLabel);
        //从对抗网络中获取生成器的输出，当然也可以用 INDArray fake =gen.output(fakeIn);
        //但这样得先把gan中的参数复制到生成器中，这样比较麻烦，浪费时间
        INDArray fakeImage = gan.activateSelectedLayers(0, generator.getLayers().length - 1, combinedLatentDim);
        //真实数据集
        DataSet realSet = new DataSet(realFeature, Nd4j.zeros(batchSize, 1));
        //生成数据集
        DataSet fakeSet = new DataSet(fakeImage, Nd4j.ones(batchSize, 1));
        //混合真实数据集和生成数据集
        DataSet combinedDataSet = DataSet.merge(Arrays.asList(realSet, fakeSet));
        //训练判别网络
        discriminator.fit(combinedDataSet);
        return combinedLatentDim;
    }

    private static void trainGan(MultiLayerNetwork generator, MultiLayerNetwork discriminator, MultiLayerNetwork gan, INDArray realLabel, int batchSize) {
        //更新对抗网络中的判别器参数
        updateDiscriminatorInGan(generator, discriminator, gan);

        INDArray noiseLatentDim= Nd4j.rand(new int[] { batchSize, 100});

        INDArray combinedNoiseLatentDim=Nd4j.concat(-1,noiseLatentDim,realLabel);
        //添加噪声数据集,即把假特征数据的标签设为真，这样的话可以训练生成器生成尽量接近于真实的数据
        DataSet noiseDataSet=new DataSet(combinedNoiseLatentDim, Nd4j.zeros(batchSize, 1));
        //训练对抗网络
        gan.fit(noiseDataSet);
    }

    private static Layer[] genLayers() {
        return new Layer[] {
                new DenseLayer.Builder().nIn(GENERATOR_INPUT_SIZE).nOut(256).weightInit(WeightInit.NORMAL).build(),
                new ActivationLayer.Builder(new ActivationLReLU(0.2)).build(),
                new DenseLayer.Builder().nIn(256).nOut(512).build(),
                new ActivationLayer.Builder(new ActivationLReLU(0.2)).build(),
                new DenseLayer.Builder().nIn(512).nOut(1024).build(),
                new ActivationLayer.Builder(new ActivationLReLU(0.2)).build(),
                new DenseLayer.Builder().nIn(1024).nOut(DISCRIMINATOR_INPUT_SIZE).activation(Activation.TANH).build()
        };
    }

    /**
     * Returns a network config that takes in a 10x10 random number and produces a 28x28 grayscale image.
     *
     * @return config
     */
    private static MultiLayerConfiguration buildGenerator() {
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(42)
                .updater(UPDATER)
                .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
                .gradientNormalizationThreshold(GRADIENT_THRESHOLD)
                //.l2(L2)
                .weightInit(WeightInit.XAVIER)
                .activation(Activation.IDENTITY)
                .list(genLayers())
                .build();

        return conf;
    }

    private static Layer[] disLayers(IUpdater updater) {
        return new Layer[] {
                //10是标签数量
                new DenseLayer.Builder().nIn(DISCRIMINATOR_INPUT_SIZE).nOut(1024).updater(updater).build(),
                new ActivationLayer.Builder(new ActivationLReLU(0.2)).build(),
                new DropoutLayer.Builder(1 - 0.5).build(),
                new DenseLayer.Builder().nIn(1024).nOut(512).updater(updater).build(),
                new ActivationLayer.Builder(new ActivationLReLU(0.2)).build(),
                new DropoutLayer.Builder(1 - 0.5).build(),
                new DenseLayer.Builder().nIn(512).nOut(256).updater(updater).build(),
                new ActivationLayer.Builder(new ActivationLReLU(0.2)).build(),
                new DropoutLayer.Builder(1 - 0.5).build(),
                new OutputLayer.Builder(LossFunctions.LossFunction.XENT).nIn(256).nOut(1).activation(Activation.SIGMOID).updater(updater).build()
        };
    }

    private static MultiLayerConfiguration buildDiscriminator() {
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(42)
                .updater(UPDATER)
                .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
                .gradientNormalizationThreshold(GRADIENT_THRESHOLD)
                //.l2(L2)
                .weightInit(WeightInit.XAVIER)
                .activation(Activation.IDENTITY)
                .list(disLayers(UPDATER))
               
                .build();

        return conf;
    }

    private static MultiLayerConfiguration buildGan() {
        Layer[] genLayers = genLayers();
        
        //判别器层学习率设为0，即冻结GAN中的判别器，让它不进行训练
        Layer[] disLayers = disLayers(UPDATER_ZERO); 
        Layer[] layers = ArrayUtils.addAll(genLayers, disLayers);

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(42)
                .updater(UPDATER)
                .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
                .gradientNormalizationThreshold(GRADIENT_THRESHOLD)
                //.l2(L2)
                .weightInit(WeightInit.XAVIER)
                .activation(Activation.IDENTITY)
                .list(layers)
                .build();

        return conf;
    }

 

    /**
     * 从GAN把参数复制到生成器和判别器
     * @param generator
     * @param discriminator
     * @param gan
     */
    private static void copyParamsToGeneratorAndDiscriminatorFromGan(MultiLayerNetwork generator, MultiLayerNetwork discriminator, MultiLayerNetwork gan) {
        int genLayerCount = generator.getLayers().length;
        for (int i = 0; i < gan.getLayers().length; i++) {
            if (i < genLayerCount) {
                generator.getLayer(i).setParams(gan.getLayer(i).params());
            } else {
                discriminator.getLayer(i - genLayerCount).setParams(gan.getLayer(i).params());
            }
        }
    }



    /**
     * 更新对抗网络中的判别器
     * generator, discriminator, gan
     * @param generator
     * @param discriminator
     * @param gan
     */
    private static void updateDiscriminatorInGan(MultiLayerNetwork generator, MultiLayerNetwork discriminator, MultiLayerNetwork gan) {
        int generatorLayerCount = generator.getLayers().length;
        for (int i = generatorLayerCount; i < gan.getLayers().length; i++) {
            gan.getLayer(i).setParams(discriminator.getLayer(i - generatorLayerCount).params());
        }
    }

    private static void visualize(INDArray[] samples) {
        if (frame == null) {
            frame = new JFrame();
            frame.setTitle("Viz");
            frame.setDefaultCloseOperation(WindowConstants.DISPOSE_ON_CLOSE);
            frame.setLayout(new BorderLayout());

            panel = new JPanel();

            panel.setLayout(new GridLayout(samples.length / 3, 1, 8, 8));
            frame.add(panel, BorderLayout.CENTER);
            frame.setVisible(true);
        }

        panel.removeAll();

        for (int i = 0; i < samples.length; i++) {
            panel.add(getImage(samples[i]));
        }

        frame.revalidate();
        frame.pack();
    }

    private static JLabel getImage(INDArray tensor) {
        BufferedImage bi = new BufferedImage(28, 28, BufferedImage.TYPE_BYTE_GRAY);
        for (int i = 0; i < 784; i++) {
            int pixel = (int)(((tensor.getDouble(i) + 1) * 2) * 255);
            bi.getRaster().setSample(i % 28, i / 28, 0, pixel);
        }
        ImageIcon orig = new ImageIcon(bi);
        
        Image imageScaled = orig.getImage().getScaledInstance((8 * 28), (8 * 28), Image.SCALE_REPLICATE);

        ImageIcon scaled = new ImageIcon(imageScaled);

        return new JLabel(scaled);
    }
    
  
}
