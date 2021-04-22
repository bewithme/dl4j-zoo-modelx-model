package org.freeware.dl4j.modelx.train.gan;

import org.deeplearning4j.core.storage.StatsStorage;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.optimize.listeners.PerformanceListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.model.stats.StatsListener;
import org.deeplearning4j.ui.model.storage.InMemoryStatsStorage;

public abstract class AbsGanTrainer {


    protected static void setListeners(ComputationGraph discriminator, ComputationGraph gan) {

        UIServer uiServer = UIServer.getInstance();

        StatsStorage statsStorage = new InMemoryStatsStorage();

        uiServer.attach(statsStorage);

        discriminator.setListeners(new PerformanceListener(10, true),new StatsListener(statsStorage));

        gan.setListeners(new PerformanceListener(10, true),new StatsListener(statsStorage));
    }
}
