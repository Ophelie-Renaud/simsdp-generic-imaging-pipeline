<?xml version="1.0" encoding="UTF-8"?>
<dftools:workflow errorOnWarning="true" verboseLevel="INFO" xmlns:dftools="http://net.sf.dftools">
    <dftools:scenario pluginId="org.ietr.preesm.scenario.task"/>
    <dftools:task pluginId="pisdf-srdag" taskId="PiMM2SrDAG">
        <dftools:data key="variables">
            <dftools:variable name="Consistency_Method" value="LCM"/>
        </dftools:data>
    </dftools:task>
    <dftools:task pluginId="pisdf-mapper.list" taskId="PiSDF Scheduling">
        <dftools:data key="variables">
            <dftools:variable name="Check" value="true"/>
            <dftools:variable name="Optimize synchronization" value="true"/>
            <dftools:variable name="balanceLoads" value="true"/>
            <dftools:variable name="edgeSchedType" value="Simple"/>
            <dftools:variable name="simulatorType" value="approximatelyTimed"/>
        </dftools:data>
    </dftools:task>
    <dftools:task
        pluginId="InitialisationExporterTask.identifier" taskId="Initialisation Stats exporter">
        <dftools:data key="variables"/>
    </dftools:task>
    <dftools:task pluginId="RadarExporterTask.identifier" taskId="Multicriteria Stats exporter">
        <dftools:data key="variables"/>
    </dftools:task>
    <dftools:task pluginId="InternodeExporterTask.identifier" taskId="Internode Stats exporter">
        <dftools:data key="variables">
            <dftools:variable name="Folder Path" value="/Algo/generated/top"/>
            <dftools:variable name="SimGrid AG Path" value="SimGrid/install_simgag.sh"/>
            <dftools:variable name="SimGrid Path" value="SimGrid/install_simgrid.sh"/>
        </dftools:data>
    </dftools:task>
    <dftools:dataTransfer from="PiMM2SrDAG" sourceport="PiMM"
        targetport="PiMM" to="PiSDF Scheduling"/>
    <dftools:dataTransfer from="scenario"
        sourceport="architecture" targetport="architecture" to="PiSDF Scheduling"/>
    <dftools:dataTransfer from="PiSDF Scheduling"
        sourceport="ABC" targetport="ABC" to="Initialisation Stats exporter"/>
    <dftools:dataTransfer from="PiSDF Scheduling"
        sourceport="ABC" targetport="ABC" to="Multicriteria Stats exporter"/>
    <dftools:dataTransfer from="Initialisation Stats exporter"
        sourceport="void" targetport="void" to="Internode Stats exporter"/>
    <dftools:dataTransfer from="Internode Stats exporter"
        sourceport="void" targetport="void" to="Multicriteria Stats exporter"/>
    <dftools:dataTransfer from="PiSDF Scheduling"
        sourceport="ABC" targetport="ABC" to="Internode Stats exporter"/>
    <dftools:dataTransfer from="scenario" sourceport="PiMM"
        targetport="PiMM" to="PiMM2SrDAG"/>
    <dftools:dataTransfer from="scenario" sourceport="scenario"
        targetport="scenario" to="PiSDF Scheduling"/>
</dftools:workflow>
