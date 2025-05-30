<?xml version="1.0" encoding="UTF-8"?>
<dftools:workflow xmlns:dftools="http://net.sf.dftools" errorOnWarning="false" verboseLevel="INFO">
    <dftools:scenario pluginId="org.ietr.preesm.scenario.task"/>
    <dftools:task pluginId="pisdf-mapper.list" taskId="Scheduling">
        <dftools:data key="variables">
            <dftools:variable name="Check" value="true"/>
            <dftools:variable name="Optimize synchronization" value="False"/>
            <dftools:variable name="balanceLoads" value="true"/>
            <dftools:variable name="edgeSchedType" value="Simple"/>
            <dftools:variable name="fastLocalSearchTime" value="10"/>
            <dftools:variable name="fastTime" value="100"/>
            <dftools:variable name="iterationNr" value="0"/>
            <dftools:variable name="iterationPeriod" value="0"/>
            <dftools:variable name="listType" value="optimised"/>
            <dftools:variable name="simulatorType" value="AccuratelyTimed"/>
        </dftools:data>
    </dftools:task>
    <dftools:task pluginId="org.ietr.preesm.memory.exclusiongraph.MemoryExclusionGraphBuilder" taskId="MEG Builder">
        <dftools:data key="variables">
            <dftools:variable name="Suppr Fork/Join" value="False"/>
            <dftools:variable name="Verbose" value="True"/>
        </dftools:data>
    </dftools:task>
    <dftools:task pluginId="org.ietr.preesm.memory.allocation.MemoryAllocatorTask" taskId="Memory Allocation">
        <dftools:data key="variables">
            <dftools:variable name="Allocator(s)" value="FirstFit"/>
            <dftools:variable name="Best/First Fit order" value="LargestFirst"/>
            <dftools:variable name="Data alignment" value="None"/>
            <dftools:variable name="Distribution" value="SharedOnly"/>
            <dftools:variable name="Merge broadcasts" value="True"/>
            <dftools:variable name="Nb of Shuffling Tested" value="10"/>
            <dftools:variable name="Verbose" value="True"/>
        </dftools:data>
    </dftools:task>
    <dftools:task pluginId="org.ietr.preesm.codegen.xtend.task.CodegenTask" taskId="Code Generation">
        <dftools:data key="variables">
            <dftools:variable name="Papify" value="false"/>
            <dftools:variable name="Printer" value="C"/>
        </dftools:data>
    </dftools:task>
    <dftools:task pluginId="pisdf-srdag" taskId="pisdf-srdag">
        <dftools:data key="variables">
            <dftools:variable name="Consistency_Method" value="LCM"/>
        </dftools:data>
    </dftools:task>
    <dftools:task pluginId="org.ietr.preesm.memory.bounds.MemoryBoundsEstimator" taskId="Memory Bounds Estimator">
        <dftools:data key="variables">
            <dftools:variable name="Solver" value="Heuristic"/>
            <dftools:variable name="Verbose" value="False"/>
        </dftools:data>
    </dftools:task>
    <dftools:task pluginId="org.ietr.preesm.stats.exporter.StatsExporterTask" taskId="Gantt Exporter">
        <dftools:data key="variables">
            <dftools:variable name="path" value="/Code/generated"/>
        </dftools:data>
    </dftools:task>
    <dftools:task pluginId="pisdf-mparams.setter" taskId="Mparams selection">
        <dftools:data key="variables">
            <dftools:variable name="1. Comparisons" value="T&gt;M"/>
            <dftools:variable name="2. Thresholds" value="33000000&gt;100000000"/>
            <dftools:variable name="3. Params objectives" value="&gt;"/>
            <dftools:variable name="4. Number heuristic" value="false"/>
            <dftools:variable name="5. Retry with delays" value="false"/>
            <dftools:variable name="6. Log path" value="/Code/generated"/>
            <dftools:variable name="6. Scheduler" value="homogeneousListPeriodic"/>
            <dftools:variable name="7. Clustering distance" value="0.0"/>
            <dftools:variable name="8. Log path" value="/Code/generated/"/>
        </dftools:data>
    </dftools:task>
    <dftools:task pluginId="pisdf-export.parameters" taskId="Param Export">
        <dftools:data key="variables">
            <dftools:variable name="path" value="/Code/generated"/>
        </dftools:data>
    </dftools:task>
    <dftools:task pluginId="org.ietr.preesm.plugin.mapper.plot" taskId="Display Gantt">
        <dftools:data key="variables"/>
    </dftools:task>
    <dftools:dataTransfer from="scenario" sourceport="architecture" targetport="architecture" to="Scheduling"/>
    <dftools:dataTransfer from="scenario" sourceport="scenario" targetport="scenario" to="Scheduling"/>
    <dftools:dataTransfer from="Scheduling" sourceport="DAG" targetport="DAG" to="MEG Builder"/>
    <dftools:dataTransfer from="scenario" sourceport="scenario" targetport="scenario" to="MEG Builder"/>
    <dftools:dataTransfer from="Memory Allocation" sourceport="MEGs" targetport="MEGs" to="Code Generation"/>
    <dftools:dataTransfer from="scenario" sourceport="scenario" targetport="scenario" to="Code Generation"/>
    <dftools:dataTransfer from="scenario" sourceport="architecture" targetport="architecture" to="Code Generation"/>
    <dftools:dataTransfer from="Scheduling" sourceport="DAG" targetport="DAG" to="Code Generation"/>
    <dftools:dataTransfer from="MEG Builder" sourceport="MemEx" targetport="MemEx" to="Memory Bounds Estimator"/>
    <dftools:dataTransfer from="Scheduling" sourceport="ABC" targetport="ABC" to="Gantt Exporter"/>
    <dftools:dataTransfer from="scenario" sourceport="scenario" targetport="scenario" to="Gantt Exporter"/>
    <dftools:dataTransfer from="pisdf-srdag" sourceport="PiMM" targetport="PiMM" to="Scheduling"/>
    <dftools:dataTransfer from="Mparams selection" sourceport="PiMM" targetport="PiMM" to="pisdf-srdag"/>
    <dftools:dataTransfer from="scenario" sourceport="scenario" targetport="scenario" to="Mparams selection"/>
    <dftools:dataTransfer from="scenario" sourceport="architecture" targetport="architecture" to="Mparams selection"/>
    <dftools:dataTransfer from="scenario" sourceport="PiMM" targetport="PiMM" to="Mparams selection"/>
    <dftools:dataTransfer from="Mparams selection" sourceport="PiMM" targetport="PiMM" to="Param Export"/>
    <dftools:dataTransfer from="Scheduling" sourceport="ABC" targetport="ABC" to="Display Gantt"/>
    <dftools:dataTransfer from="scenario" sourceport="scenario" targetport="scenario" to="Display Gantt"/>
    <dftools:dataTransfer from="MEG Builder" sourceport="MemEx" targetport="MemEx" to="Memory Allocation"/>
</dftools:workflow>
