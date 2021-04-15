package org.freeware.dl4j.modelx.model.gan;


import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@AllArgsConstructor
@NoArgsConstructor
public class GraphLayerItem {

    private String layerName;
    private Object layerOrVertex;
    private String[] layerInputs;
}
