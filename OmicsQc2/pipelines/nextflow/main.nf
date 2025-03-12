#!/usr/bin/env nextflow

/*
 * OmicsQc2 Nextflow工作流
 * 职责：以Nextflow形式实现组学数据处理和模型运行流程
 */

// 参数定义
params.raw_dir = "$projectDir/../../raw"
params.output_dir = "$projectDir/../../outputs"
params.processed_dir = "$projectDir/../../processed"
params.config_dir = "$projectDir/../../config"
params.model_path = "$projectDir/../../models/data/recon3d.xml"

// 打印流程信息
log.info """\
    OmicsQc2 Nextflow Pipeline
    ===========================
    原始数据目录 : ${params.raw_dir}
    处理后数据目录 : ${params.processed_dir}
    输出目录 : ${params.output_dir}
    配置目录 : ${params.config_dir}
    模型文件 : ${params.model_path}
    """
    .stripIndent()

// 工作流定义
workflow {
    // 数据预处理阶段
    transcriptomics_ch = Channel.fromPath("${params.raw_dir}/transcriptome*.csv")
    proteomics_ch = Channel.fromPath("${params.raw_dir}/proteomics*.csv")
    metabolomics_ch = Channel.fromPath("${params.raw_dir}/metabolomics*.csv")
    
    // 预处理转录组数据
    PREPROCESS_TRANSCRIPTOMICS(transcriptomics_ch)
    
    // 预处理蛋白组数据
    PREPROCESS_PROTEOMICS(proteomics_ch)
    
    // 预处理代谢组数据
    PREPROCESS_METABOLOMICS(metabolomics_ch)
    
    // 数据整合
    INTEGRATE_OMICS(
        PREPROCESS_TRANSCRIPTOMICS.out,
        PREPROCESS_PROTEOMICS.out,
        PREPROCESS_METABOLOMICS.out
    )
    
    // 运行pFBA模型
    RUN_PFBA(INTEGRATE_OMICS.out)
    
    // 准备机器学习数据
    PREPARE_ML_DATA(
        INTEGRATE_OMICS.out,
        RUN_PFBA.out
    )
}

// 流程步骤定义
process PREPROCESS_TRANSCRIPTOMICS {
    publishDir "${params.processed_dir}", mode: 'copy'
    
    input:
    path transcriptomics_file
    
    output:
    path "transcriptomics.csv"
    
    script:
    """
    python ${projectDir}/../../src/preprocessing/process_transcriptomics.py \
        --input ${transcriptomics_file} \
        --output transcriptomics.csv
    """
}

process PREPROCESS_PROTEOMICS {
    publishDir "${params.processed_dir}", mode: 'copy'
    
    input:
    path proteomics_file
    
    output:
    path "proteomics.csv"
    
    script:
    """
    python ${projectDir}/../../src/preprocessing/process_proteomics.py \
        --input ${proteomics_file} \
        --output proteomics.csv
    """
}

process PREPROCESS_METABOLOMICS {
    publishDir "${params.processed_dir}", mode: 'copy'
    
    input:
    path metabolomics_file
    
    output:
    path "metabolomics.csv"
    
    script:
    """
    python ${projectDir}/../../src/preprocessing/process_metabolomics.py \
        --input ${metabolomics_file} \
        --output metabolomics.csv
    """
}

process INTEGRATE_OMICS {
    publishDir "${params.output_dir}", mode: 'copy'
    
    input:
    path transcriptomics
    path proteomics
    path metabolomics
    
    output:
    path "integrated_omics.csv"
    
    script:
    """
    python ${projectDir}/../../src/integration/integrate_omics.py \
        --transcriptomics ${transcriptomics} \
        --proteomics ${proteomics} \
        --metabolomics ${metabolomics} \
        --output integrated_omics.csv
    """
}

process RUN_PFBA {
    publishDir "${params.output_dir}", mode: 'copy'
    
    input:
    path integrated_omics
    
    output:
    path "pfba_results.csv"
    
    script:
    """
    python ${projectDir}/../../models/pfba/run_pfba.py \
        --input ${integrated_omics} \
        --model ${params.model_path} \
        --output pfba_results.csv
    """
}

process PREPARE_ML_DATA {
    publishDir "${params.output_dir}", mode: 'copy'
    
    input:
    path integrated_omics
    path pfba_results
    
    output:
    path "machine_learning_ready.csv"
    
    script:
    """
    python ${projectDir}/../../models/ml/prepare_ml_data.py \
        --omics ${integrated_omics} \
        --flux ${pfba_results} \
        --output machine_learning_ready.csv
    """
} 