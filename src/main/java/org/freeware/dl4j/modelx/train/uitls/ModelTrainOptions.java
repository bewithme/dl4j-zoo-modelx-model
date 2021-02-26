package org.freeware.dl4j.modelx.train.uitls;

import com.beust.jcommander.Parameter;
import com.beust.jcommander.Parameters;

import lombok.Data;

@Parameters(separators = "=", commandDescription = "模型训练参数")
@Data
public class ModelTrainOptions {

	 @Parameter(names = {"-batchSize"}, description ="小批量大小", required = false)
	 private Integer batchSize = 2;
	 
	 @Parameter(names = {"-epochs"}, description = "训练轮数", required = false)
	 private Integer epochs = 1;
	 
	 @Parameter(names = {"-stage"}, description = "训练阶段，有些模型需要多阶段训练 ", required = false)
	 private Integer stage = 1;
	 
	 @Parameter(names = "--help", help = true)
	 private boolean help;

	 @Parameter(names = {"-configFile"}, description = "训练配置文件", required = false)
	 private String configFile;
	 
	 
	 @Parameter(names = {"-trainDataSetPath"}, description = "训练数据路径", required = false)
	 private String trainDataSetPath;
	 
	 
	 @Parameter(names = {"-testDataSetPath"}, description = "测试数据路径", required = false)
	 private String testDataSetPath;
	 
	 @Parameter(names = {"-modelSavePath"}, description = "模型保存路径", required = false)
	 private String modelSavePath;
	 
	 
	 @Parameter(names = {"-learningRate"}, description = "学习率", required = false)
	 private Double learningRate = 0.001D;

	 @Parameter(names = {"-randomSeed"}, description = "随机种子", required = false)
	 private  Integer randomSeed=123456;
	 
}
