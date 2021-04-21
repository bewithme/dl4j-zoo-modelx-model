package org.freeware.dl4j.modelx.utils;

import lombok.AllArgsConstructor;
import lombok.Data;
import org.nd4j.linalg.api.ndarray.INDArray;

@Data
@AllArgsConstructor
public class Sample {

    private INDArray feature;

    private String label;
}
