params.input_dir = "data/raw/*.csv"
params.output_dir = "results"

workflow {
    // 数据预处理阶段
    Channel.fromPath(params.input_dir)
        | collect
        | map { file -> tuple(file.baseName, file) }
        | ( preprocess )

    // 模型训练与虚拟数据生成
    preprocess.out
        | combine
        | ( run_pFBA )
}

process preprocess {
    input:
    tuple val(name), path(file)

    output:
    path("processed/${name}.parquet"), emit: processed

    script:
    """
    python bin/preprocess.py --input $file --output processed/${name}.parquet
    """
}

process run_pFBA {
    input:
    path("processed/*.parquet")

    output:
    path("virtual_data.h5"), emit: virtual

    script:
    """
    python bin/run_model.py --input processed/ --output virtual_data.h5
    """
}
