package org.freeware.dl4j.modelx.train.yolo;


import java.io.File;
import java.io.IOException;
import java.net.URI;
import java.net.URISyntaxException;
import java.util.*;

import org.apache.commons.io.FileUtils;
import org.datavec.api.io.filters.RandomPathFilter;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.objdetect.ObjectDetectionRecordReader;
import org.datavec.image.recordreader.objdetect.impl.VocLabelProvider;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.freeware.dl4j.modelx.utils.JsonUtils;
import org.freeware.dl4j.modelx.dataset.Yolo3DataSetIterator;
import org.freeware.dl4j.modelx.model.yolo.Yolo3;
import org.freeware.dl4j.modelx.train.uitls.ModelTrainOptions;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.beust.jcommander.JCommander;



/**
 * Yolo3模型训练
 * @author xuwenfeng
 *
 */
public class Yolo3Trainer {

    private static final Logger log = LoggerFactory.getLogger(Yolo3Trainer.class);
 
    private static final String IMAGES_FOLDER="JPEGImages";
    
    private static final String ANNOTATIONS_FOLDER="Annotations";
    
    private static final String ANNOTATION_FORMAT=".xml";

    private static final int[] INPUT_SHAPE = new int[] {3, 416, 416};
   

    public static void main(String[] args) throws IOException, InterruptedException {

         
    	ModelTrainOptions modelTrainOptions=new ModelTrainOptions();
		
		JCommander jCommander = JCommander.newBuilder()
                .addObject(modelTrainOptions)
                .build();
 
        jCommander.parse(args);
 
        if (modelTrainOptions.isHelp()) {
            jCommander.usage();
            return;
        }

      	File hyperparameterFile=new File(modelTrainOptions.getConfigFile());
    	
    	if(!hyperparameterFile.exists()) {
    		
    		log.error(modelTrainOptions.getConfigFile().concat(" does not exists !"));
    		
    		return;
    	}
    	
    	String hyperparameterFileJsonStr=FileUtils.readFileToString(hyperparameterFile, "UTF-8");
		
      	log.info(hyperparameterFileJsonStr);
    	
		Yolo3Hyperparameter yoloHyperparameter= JsonUtils.jsonToObject(hyperparameterFileJsonStr, Yolo3Hyperparameter.class);
		
        File imageDir = new File(yoloHyperparameter.getDataDir(), IMAGES_FOLDER);
        
        File annotationDir = new File(yoloHyperparameter.getDataDir(),ANNOTATIONS_FOLDER);
        //删除无用的系统文件
        deleteUselessFile(annotationDir);
        //删除无用的系统文件
        deleteUselessFile(imageDir);
        
        log.info("Load data...");
        
        Random random = new Random(yoloHyperparameter.getRandomSeed());
        //创建输入分割器数组
        InputSplit[] inputSplit = getInputSplit(imageDir, random,yoloHyperparameter);
        //训练集文件分割器
        InputSplit trainDataInputSplit = inputSplit[0];
        //测试集文件分割器
        InputSplit testDataInputSplit  = inputSplit[1];
        //创建训练记录读取数据集迭代器
        MultiDataSetIterator yolo3DataSetIterator = new Yolo3DataSetIterator(yoloHyperparameter.getDataDir(),yoloHyperparameter.getBatchSize(),yoloHyperparameter.getLabels(),yoloHyperparameter.getBigBoundingBoxPriors(),yoloHyperparameter.getMediumBoundingBoxPriors(),yoloHyperparameter.getSmallBoundingBoxPriors());


       //加载已有模型，如果本地不存在，则会从远程将预训练模型下载到当前用户的 
        //.deeplearning4j/models/tiny-yolo-voc_dl4j_inference.v2.zip 目录 
        ComputationGraph pretrainedComputationGraph =null;
        
        File latestModelFile=getLatestModelFile(yoloHyperparameter);
        
        if(latestModelFile==null) {
        	 pretrainedComputationGraph = (ComputationGraph) Yolo3.builder()
                     .numClasses(yoloHyperparameter.getLabels().length)
                     .bigBoundingBoxPriors(yoloHyperparameter.getBigBoundingBoxPriors())
                     .mediumBoundingBoxPriors(yoloHyperparameter.getMediumBoundingBoxPriors())
                     .smallBoundingBoxPriors(yoloHyperparameter.getSmallBoundingBoxPriors())
                     .build().init();
        }else {
             pretrainedComputationGraph = ModelSerializer.restoreComputationGraph(latestModelFile,true);
        }

       
        ComputationGraph model=pretrainedComputationGraph;
        

        log.info("\n Model Summary \n" + model.summary());

        log.info("Train model...");
        
        //设置监听器，每次迭代打印一次得分
        model.setListeners(new ScoreIterationListener(1));
        
        int startEpoch=0;
        
        if(latestModelFile!=null) {
        	startEpoch=getLatestModelFileIndex(latestModelFile);
        }
      
        long startTime=System.currentTimeMillis();
        
        String modelSavePath=yoloHyperparameter.getModelSavePath();
        
        if(!modelSavePath.endsWith(File.separator)) {
        	modelSavePath=modelSavePath.concat(File.separator);
        }
        
        for (int i = startEpoch; i < yoloHyperparameter.getEpochs(); i++) {
        	//每轮训练开始之前将数据集重置
            yolo3DataSetIterator.reset();


            while (yolo3DataSetIterator.hasNext()){
                yolo3DataSetIterator.next();
            }
           // model.fit(yolo3DataSetIterator);
      
            //每完成一轮，保存一次模型
           // ModelSerializer.writeModel(model, modelSavePath.concat(yoloHyperparameter.getName()).concat("model.zip_")+i, true);

            log.info("*** Completed epoch {} ***", i);
        }

        long endTime=System.currentTimeMillis();
        
        log.info("*** Completed all epoches at {} mins", (endTime-startTime)/(1000*60));
       
    }

	public static InputSplit[] getInputSplit(File imageDir, Random random, Yolo3Hyperparameter yoloHyperparameter) {
		
		 //随机路径过滤器，可以写规则来过滤掉不需要的数据
        RandomPathFilter pathFilter = new RandomPathFilter(random) {
            @Override
            protected boolean accept(String name) {
            	//转换为标签文件的路径
                name = name.replace(File.separator+IMAGES_FOLDER+File.separator, 
                		File.separator+ANNOTATIONS_FOLDER+File.separator)
                		.replace(yoloHyperparameter.getImageFormat(), ANNOTATION_FORMAT);
                log.info("loading annotation:"+name);
                try {
                	//如果图片文件对应的标签文件存在，则表示此条数据可以使用
                    return new File(new URI(name)).exists();
                } catch (URISyntaxException ex) {
                    throw new RuntimeException(ex);
                }
            }
        };
		InputSplit[] inputSplit = new FileSplit(
        		 imageDir,
        		 //允许的图片格式
        		 NativeImageLoader.ALLOWED_FORMATS,random)
        		 //按9:1的比例分割数据为训练集与测试集
        		.sample(pathFilter, 0.9, 0.1);
		return inputSplit;
	}


	
	public static DataSetIterator getDataSetIterator(Yolo3Hyperparameter yolo2Hyperparameter, InputSplit inputSplit, int gridHeight, int gridWidth) throws IOException {
		
		
		 //创建训练目标检测记录读取器
        ObjectDetectionRecordReader objectDetectionRecordReader = new ObjectDetectionRecordReader(
                INPUT_SHAPE[1],
                INPUT_SHAPE[2],
                INPUT_SHAPE[0],
                gridHeight,
                gridWidth,
        		new VocLabelProvider(yolo2Hyperparameter.getDataDir()));
        
        //创建记录读取器监听器，可以在加载数据时进行相应处理
       // RecordListener recordListener=new ObjectDetectRecordListener();
        //设置读取器监听器
        //objectDetectionRecordReader.setListeners(recordListener);
        //初始化训练目标检测记录读取器
        objectDetectionRecordReader.initialize(inputSplit);
        //标签开始索引
        int labelIndexFrom=1;
        //标签结束索引
        int labelIndexTo=1;
        //是否为回归任务
        boolean regression=true;

        //创建训练记录读取数据集迭代器
        DataSetIterator recordReaderDataSetIterator = new RecordReaderDataSetIterator(
        		objectDetectionRecordReader, 
        		yolo2Hyperparameter.getBatchSize(),
        		labelIndexFrom,
        		labelIndexTo, 
        		regression);
        //设置图片预处理器，将像素值归一化到0-1之间
        recordReaderDataSetIterator.setPreProcessor(new ImagePreProcessingScaler(0, 1));
        
        return recordReaderDataSetIterator;
   	}



	private static void deleteUselessFile(File file) throws IOException {
		
		if(!file.exists()) {
			log.info("file not exists");
			return ;
		}
		
		File[] uselessFiles=file.listFiles();
        
        String uselessFilePrefix="._";
        
        for(File uselessFile:uselessFiles) {
        	if(uselessFile.getName().startsWith(uselessFilePrefix)) {
        		log.info("deleting ..."+uselessFile.getName());
        		FileUtils.deleteQuietly(uselessFile);
        	}
        }
	}
	
	/**
	 * 获取最新的模型
	 * @param yolo2Hyperparameter
	 * @return
	 */
	public static File getLatestModelFile(Yolo3Hyperparameter yolo2Hyperparameter) {
		
		File modelSavePath=new  File(yolo2Hyperparameter.getModelSavePath());
		
		File[] files=modelSavePath.listFiles();
		
		if(files==null) {
			return null;
		}
		
		List<File> fileList=new ArrayList<File>();
	
		for(File file:files) {
			if(file.getName().contains(yolo2Hyperparameter.getName())) {
				fileList.add(file);
			}
		}
		if(fileList.size()==0) {
			return null;
		}
		
		Collections.sort(fileList, new Comparator<File>() {
            public int compare(File f1, File f2) {
                long diff = f1.lastModified() - f2.lastModified();
                if (diff > 0)
                    return 1;
                else if (diff == 0)
                    return 0;
                else
                	//如果 if 中修改为 返回-1 同时此处修改为返回 1  排序就会是递减
                    return -1;
            }

            public boolean equals(Object obj) {
                return true;
            }

        });
		
		return fileList.get(0);
	}
	
	 public static int getLatestModelFileIndex(File file) {
		 
		 String[] fileNames=file.getName().split("_");
		 
		 String latestModelFileIndexStr=fileNames[fileNames.length-1];
		 
		 return Integer.parseInt(latestModelFileIndexStr);
	 }
	
	
}
